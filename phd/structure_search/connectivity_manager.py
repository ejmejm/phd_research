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
    grads=None,
    updates=None,
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


def upgd_utility(
    model: DynamicNetwork,
    buffer: Float[Array, 'batch_size buffer_size'],
    grads=None,
    updates=None,
) -> Float[Array, 'max_layers max_units_per_layer']:
    """UPGD first-order feature utility: Σ_j -(dL/dW_j * W_j) per unit.

    Equivalent to batch-averaged -(dL/da) * a where a is the pre-activation,
    derived by expanding a = Σ_j W_j * x_j and using grad_W_j = E[δ * x_j].
    """
    conn_mask = (model.input_indices >= 0).astype(jnp.float32)
    per_weight = -grads.weights * model.weights * conn_mask
    step_utility = per_weight.sum(axis=-1)
    return step_utility * model.unit_mask.astype(jnp.float32)


def si_utility(
    model: DynamicNetwork,
    buffer: Float[Array, 'batch_size buffer_size'],
    grads=None,
    updates=None,
) -> Float[Array, 'max_layers max_units_per_layer']:
    """Synaptic Intelligence utility: Σ_j -(dL/dW_j * ΔW_j) per unit.

    Instantaneous per-step contribution of each unit's incoming connections
    to loss reduction, fed into the existing EMA tracker.
    """
    conn_mask = (model.input_indices >= 0).astype(jnp.float32)
    per_weight = -grads.weights * updates.weights * conn_mask
    step_utility = per_weight.sum(axis=-1)
    return step_utility * model.unit_mask.astype(jnp.float32)


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
    output_connect_strategy: str = 'all',
) -> Tuple[DynamicNetwork, 'UnitStats', Float[Array, ''],
           Bool[Array, 'max_layers max_units_per_layer'],
           Tuple]:
    """Generate new units in inactive slots, spending from the connection budget.

    Fully vectorized: samples all candidate slots at once via vmap,
    then uses cumulative cost to determine which fit in the budget.

    Returns:
        (model, unit_stats, new_budget, gen_mask_2d, gen_info)
        gen_info = (cand_layers, cand_units, gen_mask_flat, n_out_per_unit)
        for use by assign_sparse_outgoing when output_connect_strategy='random_sparse'.
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size
    output_dim = model.output_dim
    max_fan_out = model.max_fan_out

    n_total_slots = max_layers * max_units
    max_new_units = min(max_new_units, n_total_slots)

    rng, slot_rng, sample_rng, out_rng = jax.random.split(rng, 4)

    # --- 6a: Find and shuffle inactive slots ---
    inactive = (model.unit_mask == 0)  # (max_layers, max_units)
    inactive_flat = inactive.reshape(-1)  # (max_layers * max_units,)

    # Shuffle inactive slots to front via noise + argsort
    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)  # active slots go to back
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_valid = inactive_flat[cand_flat_idx]  # True if slot is actually inactive

    # --- 6b: Per-layer available source masks ---
    hidden_layers = jnp.arange(max_layers)
    hidden_units_arr = jnp.arange(max_units)
    hidden_buf_pos = (
        input_dim
        + hidden_layers[:, None] * max_units
        + hidden_units_arr[None, :]
    )

    target_layers = jnp.arange(max_layers)[:, None, None]
    source_layers = hidden_layers[None, :, None]
    source_active = model.unit_mask[None, :, :]
    hidden_available = (source_layers < target_layers) & (source_active == 1)

    layer_available = jnp.zeros((max_layers, buffer_size), dtype=jnp.bool_)
    layer_available = layer_available.at[:, :input_dim].set(True)
    for_scatter = hidden_available.reshape(max_layers, max_layers * max_units)
    hidden_buf_flat = hidden_buf_pos.reshape(-1)
    layer_available = layer_available.at[:, hidden_buf_flat].set(for_scatter)

    # --- 6c: Vectorized input connection sampling ---
    sample_keys = jax.random.split(sample_rng, max_new_units)

    def sample_one_unit(key, cand_layer):
        """Sample input connections for one candidate unit."""
        key1, key2, key3 = jax.random.split(key, 3)

        avail = layer_available[cand_layer]
        n_available = jnp.sum(avail)

        # Use half of max incoming connections (capped by available sources)
        half_conns = jnp.maximum(max_conns // 2, 1)
        n_conns = jnp.minimum(n_available, half_conns)
        n_conns = jnp.maximum(n_conns, 1)

        shuffle_noise = jax.random.uniform(key2, (buffer_size,))
        shuffle_key = jnp.where(avail, shuffle_noise, 2.0)
        sorted_idx = jnp.argsort(shuffle_key)
        selected = sorted_idx[:max_conns]

        conn_active = jnp.arange(max_conns) < n_conns
        new_indices = jnp.where(conn_active, selected, -1).astype(jnp.int32)

        bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(n_conns, 1).astype(jnp.float32))
        new_weights = jax.random.uniform(key3, (max_conns,), minval=-bound, maxval=bound)
        new_weights = jnp.where(conn_active, new_weights, 0.0)

        # Return input cost only; output cost added separately
        return new_indices, new_weights, n_conns, n_available

    all_indices, all_weights, all_input_costs, all_n_avail = jax.vmap(
        sample_one_unit
    )(sample_keys, cand_layers)

    # Units with no available sources should not be generated
    cand_valid = cand_valid & (all_n_avail > 0)

    # --- 6c2: Output cost per unit ---
    if output_connect_strategy == 'random_sparse':
        # Always use max_fan_out // 2 outgoing connections
        max_out = max(1, max_fan_out // 2)
        n_out_per_unit = jnp.full(max_new_units, max_out, dtype=jnp.int32)
        all_costs = all_input_costs + max_out
    else:
        n_out_per_unit = jnp.full(max_new_units, output_dim, dtype=jnp.int32)
        all_costs = all_input_costs + output_dim

    # --- 6d: Determine which units fit in budget ---
    costs_if_valid = jnp.where(cand_valid, all_costs.astype(jnp.float32), 0.0)
    cumulative_cost = jnp.cumsum(costs_if_valid)
    gen_mask = cand_valid & (cumulative_cost <= budget)

    # --- 6e: Apply input connections via scatter ---
    old_indices = model.input_indices[cand_layers, cand_units]
    new_input_indices_vals = jnp.where(gen_mask[:, None], all_indices, old_indices)
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        new_input_indices_vals
    )

    old_weights = model.weights[cand_layers, cand_units]
    new_weights_vals = jnp.where(gen_mask[:, None], all_weights, old_weights)
    new_weights = model.weights.at[cand_layers, cand_units].set(new_weights_vals)

    old_umask = model.unit_mask[cand_layers, cand_units]
    new_umask_vals = jnp.where(gen_mask, 1, old_umask).astype(jnp.int32)
    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(new_umask_vals)

    old_act = model.activation_indices[cand_layers, cand_units]
    new_act_vals = jnp.where(gen_mask, 0, old_act).astype(jnp.int32)
    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        new_act_vals
    )

    # --- 6f: Output connections ---
    cand_buf_pos = (
        input_dim + cand_layers * max_units + cand_units
    )

    if output_connect_strategy == 'random_sparse':
        # Output connections handled by assign_sparse_outgoing after generation
        new_output_mask = model.output_mask
        new_output_weights = model.output_weights
    else:
        # Connect to ALL output dims (existing behavior)
        new_output_mask = model.output_mask
        new_output_weights = model.output_weights
        gen_buf_mask = jnp.zeros(buffer_size, dtype=jnp.bool_)
        gen_buf_mask = gen_buf_mask.at[cand_buf_pos].set(gen_mask)
        new_output_mask = jnp.where(
            gen_buf_mask[None, :],
            jnp.ones_like(new_output_mask),
            new_output_mask,
        )
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

    gen_info = (cand_layers, cand_units, gen_mask, n_out_per_unit)
    return model, unit_stats, new_budget, gen_mask_2d, gen_info


def full_input_generate(
    model: DynamicNetwork,
    unit_stats: 'UnitStats',
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
    output_connect_strategy: str = 'all',
) -> Tuple[DynamicNetwork, 'UnitStats', Float[Array, ''],
           Bool[Array, 'max_layers max_units_per_layer'],
           Tuple]:
    """Generate new units fully connected to all inputs with ±1 weights.

    Every generated unit connects to ALL input_dim inputs with random ±1
    weights. Output connections depend on output_connect_strategy.
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size
    output_dim = model.output_dim
    max_fan_out = model.max_fan_out

    n_total_slots = max_layers * max_units
    max_new_units = min(max_new_units, n_total_slots)

    rng, slot_rng, weight_rng, out_rng = jax.random.split(rng, 4)

    # --- Find and shuffle inactive slots (same pattern as random_generate) ---
    inactive = (model.unit_mask == 0)
    inactive_flat = inactive.reshape(-1)

    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_valid = inactive_flat[cand_flat_idx]

    # --- Compute per-unit cost and budget check ---
    if output_connect_strategy == 'random_sparse':
        max_out = max(1, max_fan_out // 2)
        n_out_per_unit = jnp.full(max_new_units, max_out, dtype=jnp.int32)
        cost_per_unit_scalar = input_dim + max_out
        n_affordable = jnp.floor(budget / cost_per_unit_scalar).astype(jnp.int32)
        cumulative_valid = jnp.cumsum(cand_valid.astype(jnp.int32))
        gen_mask = cand_valid & (cumulative_valid <= n_affordable)
    else:
        n_out_per_unit = jnp.full(max_new_units, output_dim, dtype=jnp.int32)
        cost_per_unit_scalar = input_dim + output_dim
        n_affordable = jnp.floor(budget / cost_per_unit_scalar).astype(jnp.int32)
        cumulative_valid = jnp.cumsum(cand_valid.astype(jnp.int32))
        gen_mask = cand_valid & (cumulative_valid <= n_affordable)

    # --- Build input_indices: all inputs [0, 1, ..., input_dim-1] padded to max_conns ---
    full_indices = jnp.full(max_conns, -1, dtype=jnp.int32)
    full_indices = full_indices.at[:input_dim].set(jnp.arange(input_dim, dtype=jnp.int32))

    # --- ±1 weights ---
    weight_keys = jax.random.split(weight_rng, max_new_units)
    def _sample_binary_weights(key):
        return 2.0 * jax.random.bernoulli(key, 0.5, (max_conns,)).astype(jnp.float32) - 1.0
    all_weights = jax.vmap(_sample_binary_weights)(weight_keys)
    conn_mask = (jnp.arange(max_conns) < input_dim).astype(jnp.float32)
    all_weights = all_weights * conn_mask[None, :]

    # --- Apply changes via scatter ---
    old_indices = model.input_indices[cand_layers, cand_units]
    new_input_indices_vals = jnp.where(gen_mask[:, None], full_indices[None, :], old_indices)
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        new_input_indices_vals
    )

    old_weights = model.weights[cand_layers, cand_units]
    new_weights_vals = jnp.where(gen_mask[:, None], all_weights, old_weights)
    new_weights = model.weights.at[cand_layers, cand_units].set(new_weights_vals)

    old_umask = model.unit_mask[cand_layers, cand_units]
    new_umask_vals = jnp.where(gen_mask, 1, old_umask).astype(jnp.int32)
    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(new_umask_vals)

    old_act = model.activation_indices[cand_layers, cand_units]
    new_act_vals = jnp.where(gen_mask, 0, old_act).astype(jnp.int32)
    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        new_act_vals
    )

    # --- Output connections ---
    cand_buf_pos = (
        input_dim + cand_layers * max_units + cand_units
    )

    if output_connect_strategy == 'random_sparse':
        new_output_mask = model.output_mask
        new_output_weights = model.output_weights
    else:
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
    if output_connect_strategy == 'random_sparse':
        spent = (n_generated * (input_dim + max_out)).astype(jnp.float32)
    else:
        spent = (n_generated * (input_dim + output_dim)).astype(jnp.float32)
    new_budget = budget - spent

    # Build 2D affected mask for optimizer reset
    gen_mask_2d = jnp.zeros((max_layers, max_units), dtype=jnp.bool_)
    gen_mask_2d = gen_mask_2d.at[cand_layers, cand_units].set(gen_mask)

    gen_info = (cand_layers, cand_units, gen_mask, n_out_per_unit)
    return model, unit_stats, new_budget, gen_mask_2d, gen_info


def assign_sparse_outgoing(
    model: DynamicNetwork,
    gen_info: Tuple,
    max_new_units: int,
    rng: PRNGKeyArray,
) -> DynamicNetwork:
    """Assign sparse outgoing connections (output + hidden) for generated units.

    Processes units sequentially via nested jax.lax.scan to avoid conflicting
    writes when multiple new units target the same hidden unit's empty slot.

    Args:
        model: DynamicNetwork after generation (units have input connections
            but no outgoing connections yet).
        gen_info: (cand_layers, cand_units, gen_mask_flat, n_out_per_unit)
            from random_generate/full_input_generate.
        max_new_units: Maximum number of candidate units.
        rng: PRNG key for target sampling.
    """
    cand_layers, cand_units, gen_mask_flat, n_out_per_unit = gen_info

    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    output_dim = model.output_dim
    max_fan_out = model.max_fan_out
    max_out = max(1, max_fan_out // 2)

    # Combined target pool: output neurons + hidden units
    # Pool index 0..output_dim-1 = output neurons
    # Pool index output_dim + l*max_units + u = hidden unit (l, u)
    hidden_pool_size = max_layers * max_units
    pool_size = output_dim + hidden_pool_size

    keys = jax.random.split(rng, max_new_units)

    # Carry: mutable model arrays
    input_indices = model.input_indices
    weights = model.weights
    output_mask = model.output_mask
    output_weights = model.output_weights
    unit_mask = model.unit_mask  # read-only reference for availability

    def process_one_unit(carry, idx):
        input_indices, weights, output_mask, output_weights = carry

        key = keys[idx]
        is_valid = gen_mask_flat[idx]
        source_layer = cand_layers[idx]
        source_unit = cand_units[idx]
        source_bp = input_dim + source_layer * max_units + source_unit
        n_out = n_out_per_unit[idx]

        # Build target availability mask
        # Output neurons: always available
        output_available = jnp.ones(output_dim, dtype=jnp.bool_)

        # Hidden units: active, in later layer, with at least 1 empty input slot
        has_empty = jnp.any(input_indices == -1, axis=-1)  # (max_layers, max_units)
        is_later = jnp.arange(max_layers)[:, None] > source_layer
        hidden_available = (
            has_empty & is_later & (unit_mask == 1)
        ).reshape(-1)  # (hidden_pool_size,)

        pool_available = jnp.concatenate([output_available, hidden_available])

        # Shuffle available targets to front, select first n_out
        noise = jax.random.uniform(key, (pool_size,))
        sort_key = jnp.where(pool_available, noise, 2.0)
        sorted_idx = jnp.argsort(sort_key)
        selected = sorted_idx[:max_out]  # (max_out,)
        selected_valid = (jnp.arange(max_out) < n_out) & is_valid

        # Process each selected target
        def assign_one_target(carry, j):
            input_indices, weights, output_mask, output_weights = carry
            target_pool_idx = selected[j]
            should_assign = selected_valid[j]

            is_output_target = target_pool_idx < output_dim

            # --- Output target path ---
            safe_output_idx = jnp.where(is_output_target, target_pool_idx, 0)
            out_current = output_mask[safe_output_idx, source_bp]
            out_new = jnp.where(
                should_assign & is_output_target, 1, out_current
            ).astype(jnp.int32)
            output_mask = output_mask.at[safe_output_idx, source_bp].set(out_new)

            # Zero the output weight for new connections
            ow_current = output_weights[safe_output_idx, source_bp]
            ow_new = jnp.where(
                should_assign & is_output_target, 0.0, ow_current
            )
            output_weights = output_weights.at[safe_output_idx, source_bp].set(ow_new)

            # --- Hidden target path ---
            hidden_flat_idx = target_pool_idx - output_dim
            target_layer = hidden_flat_idx // max_units
            target_unit_idx = hidden_flat_idx % max_units

            # Safe indices (when is_output_target, use 0,0 — write is no-op)
            safe_tl = jnp.where(~is_output_target, target_layer, 0)
            safe_tu = jnp.where(~is_output_target, target_unit_idx, 0)

            # Find first empty slot in target unit's input_indices
            slots = input_indices[safe_tl, safe_tu]  # (max_conns,)
            empty = slots == -1
            has_empty_slot = jnp.any(empty)
            first_empty = jnp.argmax(empty)

            should_hidden = should_assign & ~is_output_target & has_empty_slot
            idx_current = input_indices[safe_tl, safe_tu, first_empty]
            idx_new = jnp.where(should_hidden, source_bp, idx_current).astype(jnp.int32)
            input_indices = input_indices.at[safe_tl, safe_tu, first_empty].set(idx_new)

            # Weight = 0 for new hidden connections
            w_current = weights[safe_tl, safe_tu, first_empty]
            w_new = jnp.where(should_hidden, 0.0, w_current)
            weights = weights.at[safe_tl, safe_tu, first_empty].set(w_new)

            return (input_indices, weights, output_mask, output_weights), None

        (input_indices, weights, output_mask, output_weights), _ = jax.lax.scan(
            assign_one_target,
            (input_indices, weights, output_mask, output_weights),
            jnp.arange(max_out),
        )

        return (input_indices, weights, output_mask, output_weights), None

    (input_indices, weights, output_mask, output_weights), _ = jax.lax.scan(
        process_one_unit,
        (input_indices, weights, output_mask, output_weights),
        jnp.arange(max_new_units),
    )

    model = eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.output_mask, m.output_weights),
        model,
        (input_indices, weights, output_mask, output_weights),
    )
    return model


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
    output_connect_strategy: str = eqx.field(static=True)
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
        output_connect_strategy: str = 'all',
        utility_fn: Optional[Callable] = None,
        generate_fn: Optional[Callable] = None,
        utility_init_fn: Optional[Callable] = None,
        *,
        rng: PRNGKeyArray,
    ):
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.max_new_units_per_step = max_new_units_per_step
        self.output_connect_strategy = output_connect_strategy
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
        grads=None,
        updates=None,
    ) -> 'ConnectivityManager':
        """Update per-unit utility estimates and accumulate connection budget."""
        step_utility = self.utility_fn(model, buffer, grads=grads, updates=updates)

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
        model, unit_stats, _, gen_mask_2d, gen_info = self.generate_fn(
            model, unit_stats, gen_budget,
            self.max_new_units_per_step,
            init_utility, gen_rng,
            output_connect_strategy=self.output_connect_strategy,
        )

        # --- Assign sparse outgoing connections if needed ---
        if self.output_connect_strategy == 'random_sparse':
            rng, sparse_rng = jax.random.split(rng)
            model = assign_sparse_outgoing(
                model, gen_info, self.max_new_units_per_step, sparse_rng,
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
    # optax.chain wraps states in a plain tuple; NamedTuples (e.g. AdamState)
    # are also tuples, so use type() to distinguish.
    is_chained = type(core_state) is tuple
    if is_chained:
        core_state = core_state[0]

    new_fields = []
    for i, val in enumerate(core_state):
        if val is None:
            new_fields.append(val)
        elif hasattr(val, 'weights') and hasattr(val, 'output_weights'):
            # val is a filtered DynamicNetwork PyTree — has .weights and .output_weights
            # Use zeros_like to preserve dtype (e.g. Adam step counts are int32)
            new_w = jnp.where(weights_mask, jnp.zeros_like(val.weights), val.weights)
            new_ow = jnp.where(output_mask, jnp.zeros_like(val.output_weights), val.output_weights)
            new_val = eqx.tree_at(
                lambda v: (v.weights, v.output_weights), val, (new_w, new_ow),
            )
            new_fields.append(new_val)
        else:
            # Scalar or other non-parameter state (e.g. learning rate) — keep as-is
            new_fields.append(val)

    new_core = core_state.__class__(*new_fields)
    new_state = (new_core, *optimizer.state[1:]) if is_chained else new_core
    return tree_replace(optimizer, state=new_state)
