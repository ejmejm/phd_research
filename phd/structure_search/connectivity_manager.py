"""Connectivity manager for DynamicNetwork structure search.

Tracks per-unit utility and periodically prunes low-utility units,
generating new randomly-connected units. Uses a connection budget
invariant: active_connections + budget = constant.

Strategy functions (utility_fn, generate_fn, utility_init_fn) are
stored as static fields and can be swapped at construction time.
"""

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray

from phd.jax_core.optimizers import EqxOptimizer
from phd.jax_core.utils import tree_replace
from phd.structure_search.dynamic_network import (
    DynamicNetwork, build_outgoing_indices,
)


EPSILON = 1e-8
MAX_FLOAT = jnp.finfo(jnp.float32).max


# ---------------------------------------------------------------------------
# Strategy functions (module-level, swappable via static fields)
# ---------------------------------------------------------------------------

def contribution_utility(
    model: DynamicNetwork,
    buffer: Float[Array, 'batch_size buffer_size'],
) -> Float[Array, 'max_layers max_units_per_layer']:
    """Contribution utility: mean(|activation|) * sum(|outgoing_weights|).

    Outgoing weights include both hidden-to-hidden (from outgoing_weights
    cache) and hidden-to-output (from output_weights * output_mask).
    """
    # Buffer positions for all hidden unit slots: (max_layers, max_units_per_layer)
    layers = jnp.arange(model.max_layers)
    units = jnp.arange(model.max_units_per_layer)
    buf_positions = (
        model.input_dim
        + layers[:, None] * model.max_units_per_layer
        + units[None, :]
    )

    # Mean absolute activation across batch
    activations = buffer[:, buf_positions]  # (batch, max_layers, max_units)
    mean_abs_act = jnp.abs(activations).mean(axis=0)

    # Hidden-to-hidden: sum |outgoing_weights| across all consuming layers and fan_out slots
    # outgoing_weights shape: (max_layers, buffer_size, max_fan_out)
    h2h_all = jnp.abs(model.outgoing_weights).sum(axis=(0, 2))  # (buffer_size,)
    h2h = h2h_all[buf_positions]

    # Hidden-to-output: sum |output_weights * output_mask| across output dims
    h2o_all = (
        jnp.abs(model.output_weights) * model.output_mask.astype(jnp.float32)
    ).sum(axis=0)  # (buffer_size,)
    h2o = h2o_all[buf_positions]

    step_utility = mean_abs_act * (h2h + h2o)
    step_utility = step_utility * model.unit_mask.astype(jnp.float32)
    return step_utility


def median_utility_init(
    unit_stats_utility: Float[Array, 'max_layers max_units_per_layer'],
    unit_mask: Int[Array, 'max_layers max_units_per_layer'],
) -> Float[Array, '']:
    """Compute median utility across active units for initializing new units."""
    any_active = jnp.any(unit_mask.astype(jnp.bool_))
    # Replace inactive slots with NaN so they don't affect median
    active_utility = jnp.where(
        unit_mask.astype(jnp.bool_),
        unit_stats_utility,
        jnp.nan,
    )
    return jnp.where(any_active, jnp.nanmedian(active_utility), 0.0)


def random_generate(
    model: DynamicNetwork,
    unit_stats: 'UnitStats',
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
) -> Tuple[DynamicNetwork, 'UnitStats', Float[Array, ''], Bool[Array, 'max_layers max_units_per_layer']]:
    """Generate new units in inactive slots, spending from the connection budget.

    Fully vectorized: samples all candidate slots at once via vmap,
    then uses cumulative cost to determine which fit in the budget.
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size
    output_dim = model.output_dim

    rng, slot_rng, sample_rng = jax.random.split(rng, 3)

    # --- 6a: Find and shuffle inactive slots ---
    inactive = (model.unit_mask == 0)  # (max_layers, max_units)
    inactive_flat = inactive.reshape(-1)  # (max_layers * max_units,)
    n_total_slots = max_layers * max_units

    # Shuffle inactive slots to front via noise + argsort
    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)  # active slots go to back
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_valid = inactive_flat[cand_flat_idx]  # True if slot is actually inactive

    # --- 6b: Per-layer available source masks ---
    # layer_available[l, pos] = True if buffer position pos is available to layer l
    # Inputs are always available; hidden unit at (l2, u) available if l2 < l and active
    hidden_layers = jnp.arange(max_layers)  # (max_layers,)
    hidden_units_arr = jnp.arange(max_units)  # (max_units,)
    hidden_buf_pos = (
        input_dim
        + hidden_layers[:, None] * max_units
        + hidden_units_arr[None, :]
    )  # (max_layers, max_units)

    # For each target layer l, hidden layer l2 is available if l2 < l
    # shape: (max_layers_target, max_layers_source, max_units)
    target_layers = jnp.arange(max_layers)[:, None, None]
    source_layers = hidden_layers[None, :, None]
    source_active = model.unit_mask[None, :, :]  # (1, max_layers, max_units)
    hidden_available = (source_layers < target_layers) & (source_active == 1)
    # (max_layers, max_layers, max_units) -> flatten source dims to buffer positions

    # Build full layer_available: (max_layers, buffer_size)
    layer_available = jnp.zeros((max_layers, buffer_size), dtype=jnp.bool_)
    layer_available = layer_available.at[:, :input_dim].set(True)
    # Scatter hidden availability
    for_scatter = hidden_available.reshape(max_layers, max_layers * max_units)
    hidden_buf_flat = hidden_buf_pos.reshape(-1)  # (max_layers * max_units,)
    layer_available = layer_available.at[:, hidden_buf_flat].set(for_scatter)

    # --- 6c: Vectorized connection sampling ---
    sample_keys = jax.random.split(sample_rng, max_new_units)

    def sample_one_unit(key, cand_layer):
        """Sample connections for one candidate unit. Pure function for vmap."""
        key1, key2, key3 = jax.random.split(key, 3)

        avail = layer_available[cand_layer]  # (buffer_size,)
        n_available = jnp.sum(avail)

        # Number of connections: uniform from 1 to min(n_available, max_conns)
        max_possible = jnp.minimum(n_available, max_conns)
        # Clamp to at least 1 to avoid randint(1, 1) issues when no sources
        max_possible = jnp.maximum(max_possible, 1)
        n_conns = jax.random.randint(key1, (), 1, max_possible + 1)

        # Shuffle available positions to front
        shuffle_noise = jax.random.uniform(key2, (buffer_size,))
        shuffle_key = jnp.where(avail, shuffle_noise, 2.0)
        sorted_idx = jnp.argsort(shuffle_key)
        selected = sorted_idx[:max_conns]

        # Mask beyond n_conns with -1
        conn_active = jnp.arange(max_conns) < n_conns
        new_indices = jnp.where(conn_active, selected, -1).astype(jnp.int32)

        # LecunUniform weights (inlined to avoid static_argnames issues with traced n_conns)
        bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(n_conns, 1).astype(jnp.float32))
        new_weights = jax.random.uniform(key3, (max_conns,), minval=-bound, maxval=bound)
        new_weights = jnp.where(conn_active, new_weights, 0.0)

        cost = n_conns + output_dim
        return new_indices, new_weights, cost, n_available

    all_indices, all_weights, all_costs, all_n_avail = jax.vmap(
        sample_one_unit
    )(sample_keys, cand_layers)
    # all_indices: (max_new_units, max_conns)
    # all_weights: (max_new_units, max_conns)
    # all_costs: (max_new_units,)

    # Units with no available sources should not be generated
    cand_valid = cand_valid & (all_n_avail > 0)

    # --- 6d: Determine which units fit in budget ---
    costs_if_valid = jnp.where(cand_valid, all_costs.astype(jnp.float32), 0.0)
    cumulative_cost = jnp.cumsum(costs_if_valid)
    gen_mask = cand_valid & (cumulative_cost <= budget)

    # --- 6e: Apply all changes via scatter ---
    # Input indices
    old_indices = model.input_indices[cand_layers, cand_units]  # (max_new_units, max_conns)
    new_input_indices_vals = jnp.where(gen_mask[:, None], all_indices, old_indices)
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        new_input_indices_vals
    )

    # Weights
    old_weights = model.weights[cand_layers, cand_units]
    new_weights_vals = jnp.where(gen_mask[:, None], all_weights, old_weights)
    new_weights = model.weights.at[cand_layers, cand_units].set(new_weights_vals)

    # Unit mask
    old_umask = model.unit_mask[cand_layers, cand_units]
    new_umask_vals = jnp.where(gen_mask, 1, old_umask).astype(jnp.int32)
    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(new_umask_vals)

    # Activation indices (0 = ReLU)
    old_act = model.activation_indices[cand_layers, cand_units]
    new_act_vals = jnp.where(gen_mask, 0, old_act).astype(jnp.int32)
    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        new_act_vals
    )

    # Output connections: set output_mask and output_weights for generated units
    cand_buf_pos = (
        input_dim + cand_layers * max_units + cand_units
    )  # (max_new_units,)

    # For output_mask: set columns at cand_buf_pos to 1 where gen_mask is True
    # output_mask shape: (output_dim, buffer_size)
    new_output_mask = model.output_mask
    new_output_weights = model.output_weights
    # Build a (buffer_size,) mask of generated buf positions
    gen_buf_mask = jnp.zeros(buffer_size, dtype=jnp.bool_)
    gen_buf_mask = gen_buf_mask.at[cand_buf_pos].set(gen_mask)
    # Set output_mask columns to 1 where generated
    new_output_mask = jnp.where(
        gen_buf_mask[None, :],
        jnp.ones_like(new_output_mask),
        new_output_mask,
    )
    # Set output_weights columns to 0 where generated
    new_output_weights = jnp.where(
        gen_buf_mask[None, :].astype(jnp.float32),
        0.0,
        new_output_weights,
    )

    model = eqx.tree_at(
        lambda m: (
            m.input_indices, m.weights, m.unit_mask,
            m.activation_indices, m.output_mask, m.output_weights,
        ),
        model,
        (
            new_input_indices, new_weights, new_unit_mask,
            new_activation_indices, new_output_mask, new_output_weights,
        ),
    )

    # Update unit stats for generated units
    old_utility = unit_stats.utility[cand_layers, cand_units]
    new_util_vals = jnp.where(gen_mask, init_utility, old_utility)
    new_utility = unit_stats.utility.at[cand_layers, cand_units].set(new_util_vals)

    old_age = unit_stats.age[cand_layers, cand_units]
    new_age_vals = jnp.where(gen_mask, 0, old_age).astype(jnp.int32)
    new_age = unit_stats.age.at[cand_layers, cand_units].set(new_age_vals)

    unit_stats = UnitStats(
        age=new_age,
        utility=new_utility,
        accumulator=unit_stats.accumulator,
    )

    # Deduct spent connections from budget
    spent = jnp.sum(jnp.where(gen_mask, all_costs.astype(jnp.float32), 0.0))
    new_budget = budget - spent

    # Build 2D affected mask for optimizer reset
    gen_mask_2d = jnp.zeros((max_layers, max_units), dtype=jnp.bool_)
    gen_mask_2d = gen_mask_2d.at[cand_layers, cand_units].set(gen_mask)

    return model, unit_stats, new_budget, gen_mask_2d


def full_input_generate(
    model: DynamicNetwork,
    unit_stats: 'UnitStats',
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
) -> Tuple[DynamicNetwork, 'UnitStats', Float[Array, ''], Bool[Array, 'max_layers max_units_per_layer']]:
    """Generate new units fully connected to all inputs with ±1 weights.

    Every generated unit connects to ALL input_dim inputs with random ±1
    weights and to ALL outputs with weight 0. Fixed cost per unit =
    input_dim + output_dim.
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size
    output_dim = model.output_dim

    cost_per_unit = input_dim + output_dim
    n_total_slots = max_layers * max_units
    max_new_units = min(max_new_units, n_total_slots)

    rng, slot_rng, weight_rng = jax.random.split(rng, 3)

    # --- Find and shuffle inactive slots (same pattern as random_generate) ---
    inactive = (model.unit_mask == 0)  # (max_layers, max_units)
    inactive_flat = inactive.reshape(-1)

    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_valid = inactive_flat[cand_flat_idx]

    # --- Determine how many units fit in budget ---
    n_affordable = jnp.floor(budget / cost_per_unit).astype(jnp.int32)
    # Use cumulative count of valid candidates vs budget
    cumulative_valid = jnp.cumsum(cand_valid.astype(jnp.int32))
    gen_mask = cand_valid & (cumulative_valid <= n_affordable)

    # --- Build input_indices: all inputs [0, 1, ..., input_dim-1] padded to max_conns ---
    full_indices = jnp.full(max_conns, -1, dtype=jnp.int32)
    full_indices = full_indices.at[:input_dim].set(jnp.arange(input_dim, dtype=jnp.int32))

    # --- ±1 weights ---
    weight_keys = jax.random.split(weight_rng, max_new_units)
    def _sample_binary_weights(key):
        return 2.0 * jax.random.bernoulli(key, 0.5, (max_conns,)).astype(jnp.float32) - 1.0
    all_weights = jax.vmap(_sample_binary_weights)(weight_keys)  # (max_new_units, max_conns)
    # Zero out padding positions beyond input_dim
    conn_mask = (jnp.arange(max_conns) < input_dim).astype(jnp.float32)
    all_weights = all_weights * conn_mask[None, :]

    # --- Apply changes via scatter (same pattern as random_generate) ---
    # Input indices
    old_indices = model.input_indices[cand_layers, cand_units]
    new_input_indices_vals = jnp.where(gen_mask[:, None], full_indices[None, :], old_indices)
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        new_input_indices_vals
    )

    # Weights
    old_weights = model.weights[cand_layers, cand_units]
    new_weights_vals = jnp.where(gen_mask[:, None], all_weights, old_weights)
    new_weights = model.weights.at[cand_layers, cand_units].set(new_weights_vals)

    # Unit mask
    old_umask = model.unit_mask[cand_layers, cand_units]
    new_umask_vals = jnp.where(gen_mask, 1, old_umask).astype(jnp.int32)
    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(new_umask_vals)

    # Activation indices (0 = first activation in tuple, e.g. ltu)
    old_act = model.activation_indices[cand_layers, cand_units]
    new_act_vals = jnp.where(gen_mask, 0, old_act).astype(jnp.int32)
    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        new_act_vals
    )

    # Output connections
    cand_buf_pos = (
        input_dim + cand_layers * max_units + cand_units
    )
    gen_buf_mask = jnp.zeros(buffer_size, dtype=jnp.bool_)
    gen_buf_mask = gen_buf_mask.at[cand_buf_pos].set(gen_mask)
    new_output_mask = jnp.where(
        gen_buf_mask[None, :],
        jnp.ones_like(model.output_mask),
        model.output_mask,
    )
    new_output_weights = jnp.where(
        gen_buf_mask[None, :].astype(jnp.float32),
        0.0,
        model.output_weights,
    )

    model = eqx.tree_at(
        lambda m: (
            m.input_indices, m.weights, m.unit_mask,
            m.activation_indices, m.output_mask, m.output_weights,
        ),
        model,
        (
            new_input_indices, new_weights, new_unit_mask,
            new_activation_indices, new_output_mask, new_output_weights,
        ),
    )

    # Update unit stats for generated units
    old_utility = unit_stats.utility[cand_layers, cand_units]
    new_util_vals = jnp.where(gen_mask, init_utility, old_utility)
    new_utility = unit_stats.utility.at[cand_layers, cand_units].set(new_util_vals)

    old_age = unit_stats.age[cand_layers, cand_units]
    new_age_vals = jnp.where(gen_mask, 0, old_age).astype(jnp.int32)
    new_age = unit_stats.age.at[cand_layers, cand_units].set(new_age_vals)

    unit_stats = UnitStats(
        age=new_age,
        utility=new_utility,
        accumulator=unit_stats.accumulator,
    )

    # Deduct spent connections
    n_generated = jnp.sum(gen_mask.astype(jnp.int32))
    spent = (n_generated * cost_per_unit).astype(jnp.float32)
    new_budget = budget - spent

    # Build 2D affected mask for optimizer reset
    gen_mask_2d = jnp.zeros((max_layers, max_units), dtype=jnp.bool_)
    gen_mask_2d = gen_mask_2d.at[cand_layers, cand_units].set(gen_mask)

    return model, unit_stats, new_budget, gen_mask_2d


# ---------------------------------------------------------------------------
# UnitStats and ConnectivityManager
# ---------------------------------------------------------------------------

class UnitStats(eqx.Module):
    """Per-unit tracking statistics across all layers.

    Stored as padded 2D arrays matching DynamicNetwork's
    (max_layers, max_units_per_layer) layout.

    Attributes:
        age: Steps since each unit was created or last reset.
        utility: Exponential moving average of per-unit utility.
        accumulator: Prune accumulator. Grows by
            prune_rate * active_connections each step. Controls how
            many connections worth of units to prune at restructure time.
            Reset to 0 after each restructure.
    """
    age: Int[Array, 'max_layers max_units_per_layer']
    utility: Float[Array, 'max_layers max_units_per_layer']
    accumulator: Float[Array, '']


class ConnectivityManager(eqx.Module):
    """Manages unit-level utility tracking and structural pruning/generation
    for DynamicNetwork.

    Prune accumulator grows by prune_rate * active_connections each step.
    At restructure: prune units worth the accumulated connections, then
    generate new units to fill up to connection_budget total connections.

    Strategy functions are stored as static fields and can be swapped at
    construction time for different utility, generation, or initialization
    behaviors.
    """

    # Static config
    decay_rate: float = eqx.field(static=True)
    maturity_threshold: int = eqx.field(static=True)
    max_new_units_per_step: int = eqx.field(static=True)
    utility_fn: Callable = eqx.field(static=True)
    generate_fn: Callable = eqx.field(static=True)
    utility_init_fn: Callable = eqx.field(static=True)

    # Dynamic state
    prune_rate: float
    connection_budget: float
    unit_stats: UnitStats
    rng: PRNGKeyArray

    def __init__(
        self,
        model: DynamicNetwork,
        prune_rate: float = 1e-4,
        connection_budget: float = 1000.0,
        decay_rate: float = 0.99,
        maturity_threshold: int = -1,
        max_new_units_per_step: int = 512,
        utility_fn: Optional[Callable] = None,
        generate_fn: Optional[Callable] = None,
        utility_init_fn: Optional[Callable] = None,
        *,
        rng: PRNGKeyArray,
    ):
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.max_new_units_per_step = max_new_units_per_step
        self.prune_rate = prune_rate
        self.connection_budget = connection_budget
        self.rng = rng

        self.utility_fn = utility_fn if utility_fn is not None else contribution_utility
        self.generate_fn = generate_fn if generate_fn is not None else random_generate
        self.utility_init_fn = utility_init_fn if utility_init_fn is not None else median_utility_init

        shape = (model.max_layers, model.max_units_per_layer)
        self.unit_stats = UnitStats(
            age=jnp.zeros(shape, dtype=jnp.int32),
            utility=jnp.zeros(shape, dtype=jnp.float32),
            accumulator=jnp.array(0.0),
        )

    def update_stats(
        self,
        model: DynamicNetwork,
        buffer: Float[Array, 'batch_size buffer_size'],
    ) -> 'ConnectivityManager':
        """Update per-unit utility estimates and accumulate connection budget."""
        step_utility = self.utility_fn(model, buffer)

        # EMA update
        new_utility = (
            (1 - self.decay_rate) * step_utility
            + self.decay_rate * self.unit_stats.utility
        )
        new_utility = new_utility * model.unit_mask.astype(jnp.float32)

        # Age increment for active units
        new_age = (self.unit_stats.age + 1) * model.unit_mask

        # Prune accumulator: grows by prune_rate * active_connections per step
        active_conns = (
            jnp.sum(model.input_indices >= 0)
            + jnp.sum(model.output_mask)
        ).astype(jnp.float32)
        new_acc = (
            self.unit_stats.accumulator
            + self.prune_rate * active_conns
        )

        new_stats = UnitStats(
            age=new_age,
            utility=new_utility,
            accumulator=new_acc,
        )
        return tree_replace(self, unit_stats=new_stats)

    def modify_structure(
        self,
        model: DynamicNetwork,
        optimizer: EqxOptimizer,
        *,
        rng: PRNGKeyArray,
    ) -> Tuple['ConnectivityManager', DynamicNetwork, EqxOptimizer,
               Int[Array, 'max_layers'], Int[Array, 'max_layers']]:
        """Prune lowest-utility units and generate new units to fill connection_budget.

        Returns:
            (tracker, model, optimizer, pruned_per_layer, generated_per_layer)
        """
        rng, prune_rng, gen_rng = jax.random.split(rng, 3)
        output_dim = model.output_dim

        # --- Compute initial utility for new units ---
        init_utility = self.utility_init_fn(
            self.unit_stats.utility, model.unit_mask,
        )

        # --- Prune mask ---
        prune_mask, n_pruned = self._make_prune_mask(
            model, rng=prune_rng,
        )

        # --- Count freed connections before clearing ---
        # Incoming connections of pruned units: count indices >= 0 in pruned slots
        pruned_conn_valid = (model.input_indices >= 0) & prune_mask[:, :, None]
        n_freed_incoming = jnp.sum(pruned_conn_valid).astype(jnp.float32)

        # Output connections of pruned units
        buf_positions = _unit_buf_positions(model)  # (max_layers, max_units)
        pruned_buf_mask = _prune_mask_to_buf_mask(prune_mask, buf_positions, model.buffer_size)
        n_freed_output = jnp.sum(
            model.output_mask * pruned_buf_mask[None, :].astype(jnp.int32)
        ).astype(jnp.float32)

        n_freed = n_freed_incoming + n_freed_output

        # --- Batch prune ---
        prune_3d = prune_mask[:, :, None]
        new_unit_mask = jnp.where(prune_mask, 0, model.unit_mask).astype(jnp.int32)
        new_weights = jnp.where(prune_3d, 0.0, model.weights)
        new_input_indices = jnp.where(prune_3d, -1, model.input_indices).astype(jnp.int32)

        pruned_buf_2d = pruned_buf_mask[None, :]  # (1, buffer_size)
        new_output_mask = (
            model.output_mask * (1 - pruned_buf_2d.astype(jnp.int32))
        )
        new_output_weights = (
            model.output_weights * (1 - pruned_buf_2d.astype(jnp.float32))
        )

        model = eqx.tree_at(
            lambda m: (
                m.unit_mask, m.weights, m.input_indices,
                m.output_mask, m.output_weights,
            ),
            model,
            (
                new_unit_mask, new_weights, new_input_indices,
                new_output_mask, new_output_weights,
            ),
        )

        # --- Reset stats for pruned units ---
        new_age = jnp.where(prune_mask, 0, self.unit_stats.age).astype(jnp.int32)
        new_utility = jnp.where(prune_mask, 0.0, self.unit_stats.utility)

        # --- Generation budget: fill up to connection_budget ---
        active_conns_after_prune = (
            jnp.sum(model.input_indices >= 0)
            + jnp.sum(model.output_mask)
        ).astype(jnp.float32)
        gen_budget = jnp.maximum(0.0, self.connection_budget - active_conns_after_prune)

        unit_stats = UnitStats(
            age=new_age,
            utility=new_utility,
            accumulator=jnp.array(0.0),  # Reset accumulator after pruning
        )

        # --- Generate new units ---
        model, unit_stats, _, gen_mask_2d = self.generate_fn(
            model, unit_stats, gen_budget,
            self.max_new_units_per_step,
            init_utility, gen_rng,
        )

        # --- Rebuild outgoing indices ---
        model = build_outgoing_indices(model)

        # --- Reset optimizer state ---
        # Affected mask: pruned OR generated units
        affected_mask = prune_mask | gen_mask_2d
        optimizer = _reset_optimizer_state(optimizer, model, affected_mask)

        # --- Per-layer prune/gen counts ---
        pruned_per_layer = prune_mask.sum(axis=1).astype(jnp.int32)
        generated_per_layer = gen_mask_2d.sum(axis=1).astype(jnp.int32)

        new_manager = tree_replace(self, unit_stats=unit_stats, rng=rng)
        return new_manager, model, optimizer, pruned_per_layer, generated_per_layer

    def _make_prune_mask(
        self,
        model: DynamicNetwork,
        *,
        rng: PRNGKeyArray,
    ) -> Tuple[Bool[Array, 'max_layers max_units_per_layer'], Int[Array, '']]:
        """Determine which units to prune based on utility ranking and connection budget."""
        unit_mask = model.unit_mask
        output_dim = model.output_dim
        budget = self.unit_stats.accumulator

        # Eligibility: active and mature
        if self.maturity_threshold > 0:
            eligible = (unit_mask == 1) & (self.unit_stats.age > self.maturity_threshold)
        else:
            eligible = (unit_mask == 1)

        n_eligible = jnp.sum(eligible)

        # Conservative estimate: each pruned unit frees at least (1 + output_dim) connections
        min_unit_cost = 1 + output_dim
        max_prunes_from_budget = (budget / min_unit_cost).astype(jnp.int32)
        max_prunes_from_budget = jnp.maximum(max_prunes_from_budget, 0)
        n_to_prune = jnp.minimum(max_prunes_from_budget, n_eligible)

        # Perturb utility for tie-breaking
        perturbed_utility = self.unit_stats.utility + jax.random.uniform(
            rng, self.unit_stats.utility.shape, minval=-EPSILON, maxval=EPSILON,
        )

        # Flatten for argsort
        flat_eligible = eligible.reshape(-1)
        flat_utility = perturbed_utility.reshape(-1)
        filtered_utility = jnp.where(flat_eligible, flat_utility, jnp.inf)

        # Find the n_to_prune lowest utilities
        ranking = jnp.argsort(filtered_utility)
        threshold = filtered_utility[ranking[n_to_prune]]
        threshold = jnp.minimum(threshold, MAX_FLOAT)

        prune_mask_flat = (filtered_utility < threshold) & flat_eligible
        prune_mask = prune_mask_flat.reshape(unit_mask.shape)

        return prune_mask, n_to_prune


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _unit_buf_positions(model: DynamicNetwork) -> Int[Array, 'max_layers max_units_per_layer']:
    """Compute buffer positions for all hidden unit slots."""
    layers = jnp.arange(model.max_layers)
    units = jnp.arange(model.max_units_per_layer)
    return model.input_dim + layers[:, None] * model.max_units_per_layer + units[None, :]


def _prune_mask_to_buf_mask(
    prune_mask: Bool[Array, 'max_layers max_units_per_layer'],
    buf_positions: Int[Array, 'max_layers max_units_per_layer'],
    buffer_size: int,
) -> Bool[Array, 'buffer_size']:
    """Convert 2D prune mask to 1D buffer position mask."""
    buf_mask = jnp.zeros(buffer_size, dtype=jnp.bool_)
    return buf_mask.at[buf_positions].set(prune_mask)



def _reset_optimizer_state(
    optimizer: EqxOptimizer,
    model: DynamicNetwork,
    affected_mask: Bool[Array, 'max_layers max_units_per_layer'],
) -> EqxOptimizer:
    """Reset optimizer state for weights associated with affected units.

    Zeros momentum/variance for:
    - weights[layer, unit_idx, :] (incoming connections)
    - output_weights[:, buf_pos] (output connections)
    """
    buf_positions = _unit_buf_positions(model)
    buf_mask = _prune_mask_to_buf_mask(affected_mask, buf_positions, model.buffer_size)

    weights_mask = affected_mask[:, :, None]  # (max_layers, max_units, 1)
    output_mask = buf_mask[None, :]  # (1, buffer_size)

    core_state = optimizer.state
    is_chained = isinstance(core_state, tuple)
    if is_chained:
        core_state = core_state[0]

    new_fields = []
    for i, val in enumerate(core_state):
        if val is None or jnp.isscalar(val):
            new_fields.append(val)
        else:
            # val is a filtered DynamicNetwork PyTree — has .weights and .output_weights
            new_w = jnp.where(weights_mask, 0.0, val.weights)
            new_ow = jnp.where(output_mask, 0.0, val.output_weights)
            new_val = eqx.tree_at(
                lambda v: (v.weights, v.output_weights), val, (new_w, new_ow),
            )
            new_fields.append(new_val)

    new_core = core_state.__class__(*new_fields)
    new_state = (new_core, *optimizer.state[1:]) if is_chained else new_core
    return tree_replace(optimizer, state=new_state)
