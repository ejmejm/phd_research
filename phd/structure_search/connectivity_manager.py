"""Connectivity manager for DynamicNetwork structure search.

Tracks per-unit utility and periodically prunes low-utility units,
replacing them with new randomly-connected units. Modeled after
CBPTracker from feature_search, adapted for sparse padded connectivity.

Budget is connection-based with two-pool accounting:
    replacement_accumulator: prune authorization (grows via replace_rate).
    connection_pool: freed connections available for generation.
The invariant connection_pool + n_active_connections = constant holds
during modify_structure. Surplus freed connections carry forward across
cycles, so the network maintains exactly constant total connections
(modulo the minimum unit cost remainder).

Each new unit gets a random number of incoming connections in
[1, max_connections_per_unit] and connects to all output dimensions
with weight 0.

Fan-out safety: when sampling input connections for a new unit at
layer l, sources that have already reached max_fan_out outgoing
connections to layer l are excluded. This ensures build_outgoing_indices
can represent all connections, keeping the backward pass accurate.

All operations use fixed-size arrays with masking for jittability.
Dynamic counts are handled via scan-based budget depletion over a
fixed MAX_REPLACEMENTS allocation.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray

from phd.jax_core.optimizers import EqxOptimizer
from phd.structure_search.dynamic_network import DynamicNetwork, build_outgoing_indices


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_REPLACEMENTS = 32   # Max units replaced per modify_structure call
_EPSILON = 1e-8          # Noise scale for utility tie-breaking
_MAX_FLOAT = 1e30        # Stand-in for infinity in masked comparisons


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _unit_buffer_positions(
    input_dim: int,
    max_layers: int,
    max_units_per_layer: int,
) -> Int[Array, 'max_layers max_units_per_layer']:
    """Map each (layer, unit) slot to its buffer position.

    Unit (l, u) → input_dim + l * max_units_per_layer + u.
    """
    layer_offsets = jnp.arange(max_layers) * max_units_per_layer + input_dim
    unit_offsets = jnp.arange(max_units_per_layer)
    return layer_offsets[:, None] + unit_offsets[None, :]


def _connections_per_unit(
    model: DynamicNetwork,
) -> Int[Array, 'max_layers max_units_per_layer']:
    """Count total active connections per unit (incoming hidden + outgoing to output).

    Incoming = active entries in input_indices[l, u, :].
    Output   = active entries in output_mask[:, buf_pos] for this unit's buffer slot.
    """
    incoming = jnp.sum(model.input_indices >= 0, axis=-1)  # (L, U)
    buf_pos = _unit_buffer_positions(
        model.input_dim, model.max_layers, model.max_units_per_layer,
    )
    output = model.output_mask[:, buf_pos].sum(axis=0).astype(jnp.int32)  # (L, U)
    return incoming + output


def _compute_fan_out(
    model: DynamicNetwork,
) -> Int[Array, 'max_layers buffer_size']:
    """Count live outgoing connections per buffer position per consuming layer.

    Only counts connections from active units (unit_mask == 1), so stale
    references from pruned units don't waste fan-out capacity.
    """
    buffer_size = model.buffer_size

    def per_layer(input_indices_l, unit_mask_l):
        active = (input_indices_l >= 0) & (unit_mask_l[:, None] == 1)  # (U, C)
        flat_src = jnp.where(active, input_indices_l, buffer_size).reshape(-1)
        counts = jnp.zeros(buffer_size + 1, dtype=jnp.int32)
        counts = counts.at[flat_src].add(1)
        return counts[:buffer_size]

    return jax.vmap(per_layer)(model.input_indices, model.unit_mask)


def _n_active_hidden_connections(model: DynamicNetwork) -> Int[Array, '']:
    """Count total active connections managed by the ConnectivityManager.

    Includes hidden-layer incoming connections and hidden→output connections,
    but NOT input→output connections (which are permanent and unmanaged).
    """
    n_incoming = jnp.sum(model.input_indices >= 0)
    n_hidden_output = jnp.sum(model.output_mask[:, model.input_dim:])
    return (n_incoming + n_hidden_output).astype(jnp.int32)


def _compute_unit_utility(
    model: DynamicNetwork,
    buffer: Float[Array, 'batch_size buffer_size'],
) -> Float[Array, 'max_layers max_units_per_layer']:
    """Compute per-unit step utility: mean|activation| * sum|outgoing_weights|.

    For each active unit at (l, u):
        buf_pos = input_dim + l * max_units_per_layer + u
        mean_abs_act = mean(|buffer[:, buf_pos]|)  over batch
        h2h = sum(|outgoing_weights[:, buf_pos, :]|)  over all layers and fan-out
        h2o = sum(|output_weights[:, buf_pos]| * output_mask[:, buf_pos])  over output_dim
        utility = mean_abs_act * (h2h + h2o)

    Inactive units (unit_mask == 0) get utility = 0.
    """
    buf_pos = _unit_buffer_positions(
        model.input_dim, model.max_layers, model.max_units_per_layer,
    )  # (L, U)

    # Mean absolute activation per unit across batch
    activations = buffer[:, buf_pos]                           # (B, L, U)
    mean_abs_act = jnp.abs(activations).mean(axis=0)           # (L, U)

    # Hidden-to-hidden: sum |outgoing_weights| across consuming layers and fan-out
    # outgoing_weights shape: (L_consuming, buffer_size, max_fan_out)
    # Index with buf_pos → (L_consuming, L_unit, U_unit, fan_out)
    h2h = jnp.abs(model.outgoing_weights)[:, buf_pos, :].sum(axis=(0, 3))  # (L, U)

    # Hidden-to-output: sum |output_weights * output_mask| across output_dim
    masked_ow = jnp.abs(model.output_weights) * model.output_mask.astype(jnp.float32)
    h2o = masked_ow[:, buf_pos].sum(axis=0)                   # (L, U)

    step_utility = mean_abs_act * (h2h + h2o)
    return step_utility * model.unit_mask.astype(jnp.float32)


def _apply_pruning(
    model: DynamicNetwork,
    prune_mask: Bool[Array, 'max_layers max_units_per_layer'],
) -> DynamicNetwork:
    """Deactivate all units marked in prune_mask.

    For each pruned unit:
    - unit_mask → 0
    - input_indices → -1, weights → 0  (clear incoming connections)
    - output_mask → 0, output_weights → 0  (clear output connections)

    Downstream units that referenced the pruned unit via input_indices
    are NOT modified — stale references are harmlessly masked by unit_mask
    in the forward pass.
    """
    prune_f = prune_mask.astype(jnp.float32)
    keep_f = 1.0 - prune_f

    # Unit mask
    new_unit_mask = model.unit_mask * (1 - prune_mask.astype(jnp.int32))

    # Incoming connections: input_indices → -1, weights → 0
    new_input_indices = jnp.where(prune_mask[:, :, None], -1, model.input_indices)
    new_weights = model.weights * keep_f[:, :, None]

    # Output connections: zero columns at pruned buffer positions
    buf_pos = _unit_buffer_positions(
        model.input_dim, model.max_layers, model.max_units_per_layer,
    )
    buf_prune = jnp.zeros(model.buffer_size, dtype=jnp.float32)
    buf_prune = buf_prune.at[buf_pos].add(prune_f)
    buf_keep = 1.0 - (buf_prune > 0).astype(jnp.float32)

    new_output_mask = (model.output_mask.astype(jnp.float32) * buf_keep[None, :]).astype(jnp.int32)
    new_output_weights = model.output_weights * buf_keep[None, :]

    return eqx.tree_at(
        lambda n: (n.unit_mask, n.input_indices, n.weights,
                   n.output_mask, n.output_weights),
        model,
        (new_unit_mask, new_input_indices, new_weights,
         new_output_mask, new_output_weights),
    )


def _sample_connections(
    layer: Int[Array, ''],
    key: PRNGKeyArray,
    input_dim: int,
    max_units_per_layer: int,
    buffer_size: int,
    max_connections_per_unit: int,
    n_conns: Int[Array, ''],
    source_available: Bool[Array, 'buffer_size'],
) -> Int[Array, 'max_connections_per_unit']:
    """Sample random input connections for a unit at the given layer.

    Available sources: buffer positions before this layer AND not at
    fan-out limit for this layer. n_conns is the desired count (clamped
    to the number of available sources).

    Uses Gumbel-top-k for jittable random subset selection. Returns sorted
    connection indices padded with -1 for unused slots.
    """
    max_source = input_dim + layer * max_units_per_layer
    positions = jnp.arange(buffer_size)
    valid = (positions < max_source) & source_available
    n_valid = jnp.sum(valid)
    n_actual = jnp.minimum(n_conns, jnp.minimum(n_valid, max_connections_per_unit))

    # Gumbel-top-k: high gumbel → selected
    gumbel = jax.random.gumbel(key, (buffer_size,))
    gumbel = jnp.where(valid, gumbel, -jnp.inf)
    selected = jnp.argsort(-gumbel)[:max_connections_per_unit]

    # Sort by position value (valid positions are smaller, so come first)
    selected = jnp.sort(selected)

    # Mask: only first n_actual entries are real connections
    conn_mask = jnp.arange(max_connections_per_unit) < n_actual
    return jnp.where(conn_mask, selected, -1)


def _init_unit_weights(
    connections: Int[Array, 'max_connections_per_unit'],
    key: PRNGKeyArray,
) -> Float[Array, 'max_connections_per_unit']:
    """Initialize incoming weights for a new unit (lecun_uniform).

    Fan-in = number of active connections (>= 0). Inactive slots get weight 0.
    """
    n_active = jnp.sum(connections >= 0).astype(jnp.float32)
    bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(n_active, 1.0))
    w = jax.random.uniform(key, connections.shape, minval=-bound, maxval=bound)
    return jnp.where(connections >= 0, w, 0.0)


def _generate_replacements(
    model: DynamicNetwork,
    generation_budget: Int[Array, ''],
    *,
    rng: PRNGKeyArray,
    max_replacements: int,
) -> Tuple[DynamicNetwork, Bool[Array, 'max_layers max_units_per_layer'], Int[Array, '']]:
    """Generate new units in random inactive slots within a connection budget.

    Each new unit:
    - Gets a random number of incoming connections in [1, max_connections_per_unit],
      sampled from sources that haven't reached their fan-out limit.
    - Connects to all output dimensions with weight 0 (cost = output_dim).
    - Total cost = actual_incoming + output_dim.

    Units are generated greedily via scan: each candidate is created if
    its estimated cost fits in the remaining budget, otherwise skipped.

    Returns:
        (model, generation_mask, connections_consumed)
    """
    slot_key, count_key, conn_key, weight_key = jax.random.split(rng, 4)
    L = model.max_layers
    U = model.max_units_per_layer
    C = model.max_connections_per_unit
    D = model.output_dim
    M = max_replacements

    # --- Fan-out availability (post-pruning) ---
    fan_out = _compute_fan_out(model)                    # (L, buffer_size)
    source_available = fan_out < model.max_fan_out       # (L, buffer_size)

    # --- Pick target slots: Gumbel-shuffle inactive positions ---
    inactive = (model.unit_mask == 0)
    noise = jax.random.gumbel(slot_key, inactive.shape)
    noise = jnp.where(inactive, noise, -jnp.inf)
    flat_order = jnp.argsort(-noise.reshape(-1))[:M]     # top M inactive
    target_layers = flat_order // U                       # (M,)
    target_slots = flat_order % U                         # (M,)
    flat_idx = target_layers * U + target_slots           # (M,) into (L*U,)

    # --- Pre-sample random connection counts in [1, max_connections] ---
    sampled_n_conns = jax.random.randint(count_key, (M,), 1, C + 1)
    estimated_cost = sampled_n_conns + D  # incoming + output connections

    # --- Determine which units fit via greedy scan over budget ---
    # Scan processes candidates in order; if a candidate is too expensive
    # it's skipped and the budget remains for cheaper candidates later.
    def budget_step(remaining, cost):
        can_afford = cost <= remaining
        new_remaining = jnp.where(can_afford, remaining - cost, remaining)
        return new_remaining, can_afford

    _, active_mask = jax.lax.scan(budget_step, generation_budget, estimated_cost)
    # active_mask: (M,) bool — which candidates to actually create

    # --- Sample connections per unit (vmapped, with fan-out constraint) ---
    conn_keys = jax.random.split(conn_key, M)

    def sample_one(layer, n_conns, key):
        avail = source_available[layer]  # (buffer_size,) for this consuming layer
        return _sample_connections(
            layer, key, model.input_dim, U, model.buffer_size, C,
            n_conns, avail,
        )

    all_connections = jax.vmap(sample_one)(
        target_layers, sampled_n_conns, conn_keys,
    )  # (M, C)

    # --- Initialize weights per unit (vmapped) ---
    weight_keys = jax.random.split(weight_key, M)
    all_weights = jax.vmap(_init_unit_weights)(all_connections, weight_keys)  # (M, C)

    # --- Compute actual connections consumed ---
    actual_incoming = jnp.sum(all_connections >= 0, axis=1)  # (M,)
    actual_cost = (actual_incoming + D) * active_mask.astype(jnp.int32)
    connections_consumed = jnp.sum(actual_cost)

    # --- Scatter into model arrays ---
    # Inactive entries (active_mask=False) write back original values (no-op).

    # unit_mask → 1
    flat_mask = model.unit_mask.reshape(-1)
    flat_mask = flat_mask.at[flat_idx].set(
        jnp.where(active_mask, 1, flat_mask[flat_idx]),
    )
    new_unit_mask = flat_mask.reshape(L, U)

    # input_indices → sampled connections
    flat_ii = model.input_indices.reshape(-1, C)
    flat_ii = flat_ii.at[flat_idx].set(
        jnp.where(active_mask[:, None], all_connections, flat_ii[flat_idx]),
    )
    new_input_indices = flat_ii.reshape(L, U, C)

    # weights → initialized weights
    flat_w = model.weights.reshape(-1, C)
    flat_w = flat_w.at[flat_idx].set(
        jnp.where(active_mask[:, None], all_weights, flat_w[flat_idx]),
    )
    new_weights = flat_w.reshape(L, U, C)

    # activation_indices → 0 (first activation function)
    flat_ai = model.activation_indices.reshape(-1)
    flat_ai = flat_ai.at[flat_idx].set(
        jnp.where(active_mask, 0, flat_ai[flat_idx]),
    )
    new_activation_indices = flat_ai.reshape(L, U)

    # output_mask → 1 for all output dims at target buffer positions
    target_buf_pos = model.input_dim + target_layers * U + target_slots  # (M,)
    active_mask_f = active_mask.astype(jnp.float32)

    new_output_mask = model.output_mask.at[:, target_buf_pos].set(
        jnp.where(active_mask[None, :], 1, model.output_mask[:, target_buf_pos]),
    )

    # output_weights → 0 at target positions
    new_output_weights = model.output_weights.at[:, target_buf_pos].set(
        jnp.where(active_mask_f[None, :], 0.0, model.output_weights[:, target_buf_pos]),
    )

    model = eqx.tree_at(
        lambda n: (n.unit_mask, n.input_indices, n.weights,
                   n.activation_indices, n.output_mask, n.output_weights),
        model,
        (new_unit_mask, new_input_indices, new_weights,
         new_activation_indices, new_output_mask, new_output_weights),
    )

    # Build generation mask (which slots received new units)
    flat_gen = jnp.zeros(L * U, dtype=jnp.bool_)
    flat_gen = flat_gen.at[flat_idx].set(active_mask)
    gen_mask = flat_gen.reshape(L, U)

    return model, gen_mask, connections_consumed


def _reset_optimizer_state(
    optimizer: EqxOptimizer,
    model: DynamicNetwork,
    reset_mask: Bool[Array, 'max_layers max_units_per_layer'],
) -> EqxOptimizer:
    """Zero optimizer state for weights associated with reset_mask units.

    Walks all leaves in the optimizer state. Any array matching the shape of
    model.weights is multiplied by a keep-mask (zeroing pruned rows). Any
    array matching model.output_weights is multiplied by a keep-mask (zeroing
    pruned columns). Other arrays are left unchanged.

    This handles Adam (mu, nu), IDBD (beta, h, etc.), and any other optax
    optimizer transparently.
    """
    reset_f = reset_mask.astype(jnp.float32)

    # weights: (L, U, C) — zero entire unit rows
    weights_keep = jnp.broadcast_to(
        1.0 - reset_f[:, :, None], model.weights.shape,
    )

    # output_weights: (D, buffer_size) — zero columns at affected buffer positions
    buf_pos = _unit_buffer_positions(
        model.input_dim, model.max_layers, model.max_units_per_layer,
    )
    buf_reset = jnp.zeros(model.buffer_size, dtype=jnp.float32)
    buf_reset = buf_reset.at[buf_pos].add(reset_f)
    ow_keep = 1.0 - (buf_reset > 0).astype(jnp.float32)
    ow_keep = ow_keep[None, :]  # (1, buffer_size) → broadcasts to (D, buffer_size)

    w_shape = model.weights.shape
    ow_shape = model.output_weights.shape

    def reset_leaf(leaf):
        if not isinstance(leaf, jnp.ndarray):
            return leaf
        if leaf.shape == w_shape:
            return leaf * weights_keep
        if leaf.shape == ow_shape:
            return leaf * ow_keep
        return leaf

    new_state = jax.tree.map(reset_leaf, optimizer.state)
    return eqx.tree_at(lambda o: o.state, optimizer, new_state)


# ---------------------------------------------------------------------------
# UnitStats
# ---------------------------------------------------------------------------

class UnitStats(eqx.Module):
    """Per-unit tracking statistics across all layers.

    Stored as padded 2D arrays matching DynamicNetwork's
    (max_layers, max_units_per_layer) layout. Inactive unit slots
    have age=0 and utility=0.

    Two-pool accounting preserves the invariant:
        connection_pool + n_active_connections = constant
    during modify_structure. Only replace_rate accumulation (in update_stats)
    changes the total system budget.

    Attributes:
        age: Steps since each unit was created or last reset.
        utility: EMA of per-unit utility (activation magnitude * outgoing weight sum).
        replacement_accumulator: Prune authorization budget. Grows by
            replace_rate * n_active_hidden_connections each step. Decremented
            by n_freed when pruning is executed.
        connection_pool: Freed connections available for generation. Incremented
            by n_freed during pruning, decremented by conns_consumed during
            generation. Carries forward surplus across cycles.
    """
    age: Int[Array, 'max_layers max_units_per_layer']
    utility: Float[Array, 'max_layers max_units_per_layer']
    replacement_accumulator: Float[Array, '']
    connection_pool: Float[Array, '']


# ---------------------------------------------------------------------------
# ConnectivityManager
# ---------------------------------------------------------------------------

class ConnectivityManager(eqx.Module):
    """Manages unit-level utility tracking and structural pruning/generation
    for DynamicNetwork.

    Lifecycle in the training loop:
        1. Every step: call update_stats(model, buffer) to update the EMA
           utility of each active unit.
        2. Every nth step: call modify_structure(model, optimizer, rng=...)
           to prune the lowest-utility mature units and generate replacements.

    Two-pool connection accounting:
        The system uses two separate pools to maintain the invariant
        connection_pool + n_active_connections = constant during
        modify_structure:

        replacement_accumulator: Prune authorization budget. Grows by
            replace_rate * n_active_hidden_connections each step.
            Decremented by n_freed when pruning is executed.

        connection_pool: Freed connections available for generation.
            Incremented by n_freed during pruning (connections move from
            the network into the pool). Decremented by conns_consumed
            during generation (connections move from the pool back into
            the network). Surplus carries forward across cycles, so freed
            connections are never lost.

        The generation budget is the full connection_pool, not just
        this cycle's n_freed. With enough inactive slots
        (max_units_per_layer > active units), generation can draw from
        the full pool to create enough smaller units to consume the
        available connections.

    Fan-out safety:
        When sampling input connections for new units, sources at their
        max_fan_out limit for the target layer are excluded. This ensures
        build_outgoing_indices can represent all connections and the
        backward pass stays accurate.

    Attributes:
        decay_rate: EMA decay for utility tracking (static).
        maturity_threshold: Min age before a unit can be pruned; -1 to disable (static).
        max_replacements: Upper bound on units replaced per call (static).
        replace_rate: Fraction of active connections to budget per step.
        unit_stats: Current per-unit statistics.
        rng: PRNG key for random operations.
    """

    # Static config (won't cause recompilation when unchanged)
    decay_rate: float = eqx.field(static=True)
    maturity_threshold: int = eqx.field(static=True)
    max_replacements: int = eqx.field(static=True)

    # Dynamic state (carried in TrainState, updated each step)
    replace_rate: float
    unit_stats: UnitStats
    rng: PRNGKeyArray

    def __init__(
        self,
        model: DynamicNetwork,
        replace_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = -1,
        max_replacements: int = _MAX_REPLACEMENTS,
        *,
        rng: PRNGKeyArray,
    ):
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.max_replacements = max_replacements
        self.replace_rate = replace_rate
        self.rng = rng

        shape = (model.max_layers, model.max_units_per_layer)
        self.unit_stats = UnitStats(
            age=jnp.zeros(shape, dtype=jnp.int32),
            utility=jnp.zeros(shape, dtype=jnp.float32),
            replacement_accumulator=jnp.array(0.0),
            connection_pool=jnp.array(0.0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_stats(
        self,
        model: DynamicNetwork,
        buffer: Float[Array, 'batch_size buffer_size'],
    ) -> 'ConnectivityManager':
        """Update per-unit utility EMA, increment ages, accumulate connection budget.

        Called every training step. Only active units (unit_mask == 1) are
        updated; inactive slots remain zeroed. The replacement accumulator
        grows by replace_rate * n_active_hidden_connections (excluding
        permanent input→output connections).
        """
        active = model.unit_mask.astype(jnp.float32)

        # Step utility via CBP formula
        step_utility = _compute_unit_utility(model, buffer)

        # EMA update (masked to active units)
        new_utility = (
            (1.0 - self.decay_rate) * step_utility
            + self.decay_rate * self.unit_stats.utility
        ) * active

        # Age: +1 for active units
        new_age = self.unit_stats.age + model.unit_mask.astype(jnp.int32)

        # Connection-based replacement budget (hidden connections only)
        n_conns = _n_active_hidden_connections(model).astype(jnp.float32)
        new_accumulator = (
            self.unit_stats.replacement_accumulator
            + self.replace_rate * n_conns
        )

        new_stats = UnitStats(
            age=new_age,
            utility=new_utility,
            replacement_accumulator=new_accumulator,
            connection_pool=self.unit_stats.connection_pool,
        )
        return eqx.tree_at(lambda s: s.unit_stats, self, new_stats)

    def modify_structure(
        self,
        model: DynamicNetwork,
        optimizer: EqxOptimizer,
        *,
        rng: PRNGKeyArray,
    ) -> Tuple['ConnectivityManager', DynamicNetwork, EqxOptimizer]:
        """Prune lowest-utility units and generate replacements.

        Two-pool accounting preserves connection_pool + n_active = constant:
            1. Prune: replacement_accumulator -= n_freed (authorization spent),
               connection_pool += n_freed (freed connections deposited).
            2. Generate: connection_pool -= conns_consumed (connections withdrawn).
        The generation budget is the full connection_pool (including surplus
        carried forward from previous cycles), so freed connections are never
        lost — they stay in the pool until generation can use them.

        Steps:
            1. Select units to prune within accumulator budget.
            2. Deactivate pruned units.
            3. Reset stats (age, utility) for pruned slots.
            4. Deposit freed connections into pool; withdraw accumulator.
            5. Generate new units funded by the full connection pool.
            6. Withdraw consumed connections from pool.
            7. Rebuild outgoing indices.
            8. Reset optimizer state for all affected positions.
        """
        prune_key, gen_key = jax.random.split(rng)

        # 1. Determine which units to prune (accumulator-budget-aware)
        prune_mask, n_freed = self._make_prune_mask(model, rng=prune_key)

        # 2. Deactivate pruned units
        model = _apply_pruning(model, prune_mask)

        # 3. Reset stats for pruned units
        new_age = jnp.where(prune_mask, 0, self.unit_stats.age)
        new_utility = jnp.where(prune_mask, 0.0, self.unit_stats.utility)

        # 4. Accumulator spends prune authorization; pool receives freed connections
        new_accumulator = (
            self.unit_stats.replacement_accumulator
            - n_freed.astype(jnp.float32)
        )
        generation_pool = (
            self.unit_stats.connection_pool
            + n_freed.astype(jnp.float32)
        )

        # 5. Generate replacements (budget = full connection pool)
        generation_budget = jnp.floor(generation_pool).astype(jnp.int32)
        model, gen_mask, conns_consumed = _generate_replacements(
            model, generation_budget, rng=gen_key,
            max_replacements=self.max_replacements,
        )

        # 6. Pool withdraws consumed connections (surplus carries forward)
        new_pool = generation_pool - conns_consumed.astype(jnp.float32)

        # 7. Rebuild outgoing indices after structural changes
        model = build_outgoing_indices(model)

        # 8. Reset optimizer state for pruned AND generated positions
        combined_mask = prune_mask | gen_mask
        optimizer = _reset_optimizer_state(optimizer, model, combined_mask)

        new_stats = UnitStats(
            age=new_age,
            utility=new_utility,
            replacement_accumulator=new_accumulator,
            connection_pool=new_pool,
        )
        new_self = eqx.tree_at(lambda s: s.unit_stats, self, new_stats)
        return new_self, model, optimizer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_prune_mask(
        self,
        model: DynamicNetwork,
        *,
        rng: PRNGKeyArray,
    ) -> Tuple[Bool[Array, 'max_layers max_units_per_layer'], Int[Array, '']]:
        """Select lowest-utility units for pruning within connection budget.

        Budget = floor(accumulator) connections. Units are sorted by utility
        (ascending) and greedily accepted: each unit is pruned if its
        connection count fits in the remaining budget, otherwise skipped.
        This avoids a single expensive unit blocking cheaper units behind it.

        Returns:
            (prune_mask, n_freed_connections)
        """
        stats = self.unit_stats
        unit_mask = model.unit_mask
        active = unit_mask.astype(jnp.bool_)

        # Eligibility: active and mature
        if self.maturity_threshold > 0:
            eligible = active & (stats.age > self.maturity_threshold)
        else:
            eligible = active

        # Per-unit connection count
        conns = _connections_per_unit(model)  # (L, U)
        eligible_conns = conns * eligible.astype(jnp.int32)  # 0 for ineligible

        # Connection budget
        prune_budget = jnp.floor(stats.replacement_accumulator).astype(jnp.int32)

        # Perturb utility to break ties
        noise = jax.random.uniform(
            rng, stats.utility.shape, minval=-_EPSILON, maxval=_EPSILON,
        )
        perturbed = stats.utility + noise

        # Ineligible → very high utility (sorted to end, never pruned)
        masked_utility = jnp.where(eligible, perturbed, _MAX_FLOAT)

        # Sort by utility ascending
        flat_utility = masked_utility.reshape(-1)
        flat_conns = eligible_conns.reshape(-1)
        flat_eligible = eligible.reshape(-1)
        ranking = jnp.argsort(flat_utility)

        sorted_conns = flat_conns[ranking]
        sorted_eligible = flat_eligible[ranking]

        # Greedy scan: prune if affordable, skip if too expensive.
        # This lets cheap low-utility units be pruned even if an expensive
        # unit earlier in the ranking can't fit in the remaining budget.
        def budget_step(remaining, data):
            conn, elig = data
            can_afford = (conn <= remaining) & elig & (conn > 0)
            new_remaining = jnp.where(can_afford, remaining - conn, remaining)
            return new_remaining, can_afford

        _, can_prune = jax.lax.scan(
            budget_step, prune_budget, (sorted_conns, sorted_eligible),
        )

        # Cap at max_replacements units
        cum_pruneable = jnp.cumsum(can_prune.astype(jnp.int32))
        final_prune_sorted = can_prune & (cum_pruneable <= self.max_replacements)

        # Unsort back to (L*U) layout
        flat_prune = jnp.zeros(flat_utility.shape[0], dtype=jnp.bool_)
        flat_prune = flat_prune.at[ranking].set(final_prune_sorted)
        prune_mask = flat_prune.reshape(model.max_layers, model.max_units_per_layer)

        # Total connections freed (using full connection count, not eligible_conns,
        # since pruned units lose all their connections)
        n_freed = jnp.sum(conns * prune_mask.astype(jnp.int32))

        return prune_mask, n_freed
