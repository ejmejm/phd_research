"""Column-guided structure search experiment.

Tests whether a dynamic network starting from full random connectivity can
converge to independent per-task subnetworks and match the block-sparse
baseline — isolating loss-of-plasticity from search quality.

Column assignment
-----------------
With n_tasks tasks the network is divided into n_tasks vertical slices:
  input  j  -> column  j // (input_dim // n_tasks)
  hidden i  -> column  i // (max_units_per_layer // n_tasks)
  output j  -> column  j // (output_dim // n_tasks)

Utility (two-phase)
-------------------
  cross > 0 : utility = -cross_count          (pruned before all within-col units)
  cross == 0: utility = contribution_utility  (normal plasticity-aware selection)

Generation
----------
New unit in slot (l, i) connects only to within-column inputs/hidden/outputs.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial
from typing import Optional, Tuple

# Resolve MLflow URI before Hydra changes CWD
_mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
if _mlflow_uri.startswith('sqlite:///') and not os.path.isabs(_mlflow_uri[len('sqlite:///'):]):
    os.environ['MLFLOW_TRACKING_URI'] = (
        f'sqlite:///{os.path.abspath(_mlflow_uri[len("sqlite:///"):])}'
    )

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import (
    prepare_optimizer, set_seed, rng_from_string,
)
from phd.jax_core.utils import configure_jax, count_params, stack_pytrees, tree_replace
from phd.research_utils.logging import (
    init_experiment, init_child_runs, log_metrics, log_child_metrics,
    finish_child_runs, finish_experiment,
)
from phd.structure_search.connectivity_manager import (
    ConnectivityManager, ConnectionConnectivityManager, ConnectionStats,
    UnitStats, contribution_utility, median_utility_init,
    _unit_buf_positions, _prune_mask_to_buf_mask, _reset_optimizer_state,
    _reset_optimizer_state_connections, assign_sparse_outgoing,
)
from phd.structure_search.data import load_dataset, ParallelMNISTStream
from phd.jax_core.models import lecun_uniform
from phd.structure_search.dynamic_network import (
    DynamicNetwork, sync_outgoing_weights, build_outgoing_indices,
    init_random_dynamic_network, count_active_connections, count_active_units,
)
from phd.structure_search.metrics import StepMetrics, compute_structure_metrics


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalized_contribution_utility(
    model: DynamicNetwork,
    buffer: Float[Array, 'batch_size buffer_size'],
    grads=None,
    updates=None,
    targets=None,
    predictions=None,
) -> Float[Array, 'max_layers max_units_per_layer']:
    """Contribution utility divided by the number of active outgoing connections.

    Counts all active outgoing connections per unit (hidden-to-hidden via
    outgoing_unit_indices, plus hidden-to-output via output_mask) and
    divides the raw contribution utility by that count (clamped to >= 1).
    """
    contrib = contribution_utility(
        model, buffer, grads=grads, updates=updates,
        targets=targets, predictions=predictions,
    )

    layers = jnp.arange(model.max_layers)
    units = jnp.arange(model.max_units_per_layer)
    buf_positions = (
        model.input_dim
        + layers[:, None] * model.max_units_per_layer
        + units[None, :]
    )  # (max_layers, max_units)

    # Hidden-to-hidden outgoing count: sum over all layers and fan_out slots
    h2h_count = (model.outgoing_unit_indices >= 0).sum(axis=(0, 2))[buf_positions]

    # Hidden-to-output outgoing count
    h2o_count = model.output_mask[:, buf_positions].astype(jnp.int32).sum(axis=0)

    n_out = jnp.maximum((h2h_count + h2o_count).astype(jnp.float32), 1.0)
    return contrib / n_out


def column_utility(
    model: DynamicNetwork,
    buffer: Float[Array, 'batch_size buffer_size'],
    grads=None,
    updates=None,
    targets=None,
    predictions=None,
    *,
    n_tasks: int,
) -> Float[Array, 'max_layers max_units_per_layer']:
    """Two-phase utility based on column structure.

    Units with any cross-column connections: utility = -(cross connection count).
    Units with only within-column connections: utility = contribution_utility.
    Inactive units: utility = 0.
    """
    input_dim = model.input_dim
    output_dim = model.output_dim
    max_layers = model.max_layers
    max_units = model.max_units_per_layer

    col_size = max_units // n_tasks
    input_col_size = input_dim // n_tasks
    out_col_size = output_dim // n_tasks

    # Buffer positions for all hidden units: (max_layers, max_units)
    layers = jnp.arange(max_layers)
    units_idx = jnp.arange(max_units)
    buf_positions = input_dim + layers[:, None] * max_units + units_idx[None, :]

    # Column of each unit: (max_units,)
    unit_col = units_idx // col_size

    # --- Incoming cross-column connections ---
    idx = model.input_indices  # (max_layers, max_units, max_conns)
    is_active_conn = idx >= 0
    is_input_src = (idx >= 0) & (idx < input_dim)

    safe_idx = jnp.maximum(idx, 0)
    src_col_if_input = safe_idx // input_col_size
    src_col_if_hidden = (safe_idx - input_dim) % max_units // col_size
    src_col = jnp.where(is_input_src, src_col_if_input, src_col_if_hidden)

    # unit_col: (max_units,) → (1, max_units, 1)
    unit_col_3d = unit_col[None, :, None]
    in_cross = is_active_conn & (src_col != unit_col_3d)  # (max_layers, max_units, max_conns)
    in_cross_count = in_cross.sum(axis=-1)  # (max_layers, max_units)

    # --- Outgoing cross-column output connections ---
    # output_mask: (output_dim, buffer_size)
    out_mask_at_units = model.output_mask[:, buf_positions]      # (output_dim, max_layers, max_units)
    out_mask_at_units = jnp.transpose(out_mask_at_units, (1, 2, 0))  # (max_layers, max_units, output_dim)

    out_col_of_k = jnp.arange(output_dim) // out_col_size       # (output_dim,)
    unit_col_out = unit_col[None, :, None]                        # (1, max_units, 1)
    out_cross = out_mask_at_units.astype(jnp.bool_) & (out_col_of_k[None, None, :] != unit_col_out)
    out_cross_count = out_cross.sum(axis=-1)                      # (max_layers, max_units)

    cross_count = (in_cross_count + out_cross_count).astype(jnp.float32)

    # Contribution utility for already-converged units
    contrib = contribution_utility(
        model, buffer,
        grads=grads, updates=updates, targets=targets, predictions=predictions,
    )

    util = jnp.where(cross_count > 0, -cross_count, contrib)
    return util * model.unit_mask.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Column-aware generation
# ---------------------------------------------------------------------------

def column_assign_outgoing(
    model: DynamicNetwork,
    gen_info: tuple,
    max_new_units: int,
    rng: PRNGKeyArray,
    *,
    n_tasks: int,
) -> DynamicNetwork:
    """Push sparse outgoing connections from generated units to downstream targets.

    For each generated unit at column c, randomly samples up to max_fan_out // 2
    targets from the within-column downstream pool:
      - Output neurons in [c * out_col_size, (c+1) * out_col_size)
      - Active hidden units in later layers, same column, with an empty input slot

    Output targets: sets output_mask = 1 and zero-initialises output_weights.
    Hidden targets: scatters source buffer position into the target unit's first
    empty input_indices slot with weight = 0.

    Uses nested jax.lax.scan to process units then targets sequentially,
    avoiding write-after-write conflicts when multiple new units target the
    same existing hidden unit.
    """
    cand_layers, cand_units, gen_mask_flat, n_out_per_unit = gen_info

    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    input_dim = model.input_dim
    output_dim = model.output_dim
    max_fan_out = model.max_fan_out
    max_out = max(1, max_fan_out // 2)

    col_size = max_units // n_tasks
    out_col_size = output_dim // n_tasks

    # Pool layout: 0..output_dim-1 → output neurons;
    #              output_dim..output_dim+max_layers*max_units-1 → hidden units
    pool_size = output_dim + max_layers * max_units

    keys = jax.random.split(rng, max_new_units)

    input_indices = model.input_indices    # (max_layers, max_units, max_conns)
    weights_arr = model.weights            # (max_layers, max_units, max_conns)
    output_mask = model.output_mask        # (output_dim, buffer_size)
    output_weights = model.output_weights  # (output_dim, buffer_size)
    unit_mask = model.unit_mask            # (max_layers, max_units)

    def process_one_unit(carry, idx):
        input_indices, weights_arr, output_mask, output_weights = carry

        key = keys[idx]
        is_valid = gen_mask_flat[idx]
        source_layer = cand_layers[idx]
        source_unit = cand_units[idx]
        source_col = source_unit // col_size
        source_bp = input_dim + source_layer * max_units + source_unit
        n_out = n_out_per_unit[idx]

        # Output targets: within-column outputs only
        out_idx = jnp.arange(output_dim)
        col_out_start = source_col * out_col_size
        col_out_end = col_out_start + out_col_size
        output_available = (out_idx >= col_out_start) & (out_idx < col_out_end)

        # Hidden targets: later layer, same column, active, has an empty slot
        has_empty_slot = jnp.any(input_indices == -1, axis=-1)  # (max_layers, max_units)
        unit_col = jnp.arange(max_units) // col_size            # (max_units,)
        is_same_col = unit_col[None, :] == source_col           # (1, max_units)
        is_later = jnp.arange(max_layers)[:, None] > source_layer
        hidden_available = (
            has_empty_slot & is_later & (unit_mask == 1) & is_same_col
        ).reshape(-1)  # (max_layers * max_units,)

        pool_available = jnp.concatenate([output_available, hidden_available])

        noise = jax.random.uniform(key, (pool_size,))
        sort_key = jnp.where(pool_available, noise, 2.0)
        selected = jnp.argsort(sort_key)[:max_out]  # (max_out,)
        selected_valid = (jnp.arange(max_out) < n_out) & is_valid

        def assign_one_target(carry, j):
            input_indices, weights_arr, output_mask, output_weights = carry
            target_pool_idx = selected[j]
            should_assign = selected_valid[j] & pool_available[target_pool_idx]

            is_out = target_pool_idx < output_dim

            # Output path
            safe_out = jnp.where(is_out, target_pool_idx, 0)
            out_new = jnp.where(
                should_assign & is_out, 1, output_mask[safe_out, source_bp]
            ).astype(jnp.int32)
            output_mask = output_mask.at[safe_out, source_bp].set(out_new)
            ow_new = jnp.where(
                should_assign & is_out, 0.0, output_weights[safe_out, source_bp]
            )
            output_weights = output_weights.at[safe_out, source_bp].set(ow_new)

            # Hidden path
            hid_flat = target_pool_idx - output_dim
            tl = hid_flat // max_units
            tu = hid_flat % max_units
            safe_tl = jnp.where(~is_out, tl, 0)
            safe_tu = jnp.where(~is_out, tu, 0)

            slots = input_indices[safe_tl, safe_tu]  # (max_conns,)
            first_empty = jnp.argmax(slots == -1)
            has_slot = jnp.any(slots == -1)
            should_hid = should_assign & ~is_out & has_slot

            idx_new = jnp.where(
                should_hid, source_bp, input_indices[safe_tl, safe_tu, first_empty]
            ).astype(jnp.int32)
            input_indices = input_indices.at[safe_tl, safe_tu, first_empty].set(idx_new)
            w_new = jnp.where(should_hid, 0.0, weights_arr[safe_tl, safe_tu, first_empty])
            weights_arr = weights_arr.at[safe_tl, safe_tu, first_empty].set(w_new)

            return (input_indices, weights_arr, output_mask, output_weights), None

        (input_indices, weights_arr, output_mask, output_weights), _ = jax.lax.scan(
            assign_one_target,
            (input_indices, weights_arr, output_mask, output_weights),
            jnp.arange(max_out),
        )
        return (input_indices, weights_arr, output_mask, output_weights), None

    (input_indices, weights_arr, output_mask, output_weights), _ = jax.lax.scan(
        process_one_unit,
        (input_indices, weights_arr, output_mask, output_weights),
        jnp.arange(max_new_units),
    )
    return eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.output_mask, m.output_weights),
        model,
        (input_indices, weights_arr, output_mask, output_weights),
    )


def assign_outgoing_relaxed(
    model: DynamicNetwork,
    gen_info: tuple,
    max_new_units: int,
    rng: PRNGKeyArray,
    *,
    n_tasks: int,
    column_priority: bool = True,
) -> DynamicNetwork:
    """Push sparse outgoing connections with an expanded target pool.

    Unlike column_assign_outgoing, the pool includes ALL outputs and ALL
    later-layer active hidden units (not restricted to the source column).

    column_priority=True (variants 1+2): within-column targets are sorted
    first (sort_key in [0,1)); cross-column available targets come next
    (sort_key in [1,2)); unavailable targets are last (sort_key=3.0).

    column_priority=False (variant 3): purely random ranking over all
    available targets; no column distinction.

    n_out_per_unit in gen_info is sampled randomly (1..max_out) per unit.
    """
    cand_layers, cand_units, gen_mask_flat, n_out_per_unit = gen_info

    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    input_dim = model.input_dim
    output_dim = model.output_dim
    max_fan_out = model.max_fan_out
    max_out = max(1, max_fan_out // 2)

    col_size = max_units // n_tasks
    out_col_size = output_dim // n_tasks

    pool_size = output_dim + max_layers * max_units
    keys = jax.random.split(rng, max_new_units)

    input_indices = model.input_indices    # (max_layers, max_units, max_conns)
    weights_arr = model.weights            # (max_layers, max_units, max_conns)
    output_mask = model.output_mask        # (output_dim, buffer_size)
    output_weights = model.output_weights  # (output_dim, buffer_size)
    unit_mask = model.unit_mask            # (max_layers, max_units)

    def process_one_unit(carry, idx):
        input_indices, weights_arr, output_mask, output_weights = carry

        key = keys[idx]
        is_valid = gen_mask_flat[idx]
        source_layer = cand_layers[idx]
        source_unit = cand_units[idx]
        source_col = source_unit // col_size
        source_bp = input_dim + source_layer * max_units + source_unit
        n_out = n_out_per_unit[idx]

        # All outputs are available targets
        out_idx = jnp.arange(output_dim)
        output_available = jnp.ones(output_dim, dtype=jnp.bool_)

        # Hidden targets: later layer, active, has an empty slot (no column restriction)
        has_empty_slot = jnp.any(input_indices == -1, axis=-1)  # (max_layers, max_units)
        is_later = jnp.arange(max_layers)[:, None] > source_layer
        hidden_available = (
            has_empty_slot & is_later & (unit_mask == 1)
        ).reshape(-1)

        pool_available = jnp.concatenate([output_available, hidden_available])

        noise = jax.random.uniform(key, (pool_size,))

        if column_priority:
            col_out_start = source_col * out_col_size
            out_in_col = (out_idx >= col_out_start) & (out_idx < col_out_start + out_col_size)
            unit_col_arr = jnp.arange(max_units) // col_size
            hid_in_col = (
                (unit_col_arr[None, :] == source_col) & (unit_mask == 1)
                & has_empty_slot & is_later
            ).reshape(-1)
            pool_in_col = jnp.concatenate([out_in_col, hid_in_col])
            sort_key = jnp.where(
                pool_available & pool_in_col, noise,
                jnp.where(pool_available & ~pool_in_col, 1.0 + noise, 3.0),
            )
        else:
            sort_key = jnp.where(pool_available, noise, 2.0)

        selected = jnp.argsort(sort_key)[:max_out]
        selected_valid = (jnp.arange(max_out) < n_out) & is_valid

        def assign_one_target(carry, j):
            input_indices, weights_arr, output_mask, output_weights = carry
            target_pool_idx = selected[j]
            should_assign = selected_valid[j] & pool_available[target_pool_idx]

            is_out = target_pool_idx < output_dim

            # Output path
            safe_out = jnp.where(is_out, target_pool_idx, 0)
            out_new = jnp.where(
                should_assign & is_out, 1, output_mask[safe_out, source_bp]
            ).astype(jnp.int32)
            output_mask = output_mask.at[safe_out, source_bp].set(out_new)
            ow_new = jnp.where(
                should_assign & is_out, 0.0, output_weights[safe_out, source_bp]
            )
            output_weights = output_weights.at[safe_out, source_bp].set(ow_new)

            # Hidden path
            hid_flat = target_pool_idx - output_dim
            tl = hid_flat // max_units
            tu = hid_flat % max_units
            safe_tl = jnp.where(~is_out, tl, 0)
            safe_tu = jnp.where(~is_out, tu, 0)

            slots = input_indices[safe_tl, safe_tu]  # (max_conns,)
            first_empty = jnp.argmax(slots == -1)
            has_slot = jnp.any(slots == -1)
            should_hid = should_assign & ~is_out & has_slot

            idx_new = jnp.where(
                should_hid, source_bp, input_indices[safe_tl, safe_tu, first_empty]
            ).astype(jnp.int32)
            input_indices = input_indices.at[safe_tl, safe_tu, first_empty].set(idx_new)
            w_new = jnp.where(should_hid, 0.0, weights_arr[safe_tl, safe_tu, first_empty])
            weights_arr = weights_arr.at[safe_tl, safe_tu, first_empty].set(w_new)

            return (input_indices, weights_arr, output_mask, output_weights), None

        (input_indices, weights_arr, output_mask, output_weights), _ = jax.lax.scan(
            assign_one_target,
            (input_indices, weights_arr, output_mask, output_weights),
            jnp.arange(max_out),
        )
        return (input_indices, weights_arr, output_mask, output_weights), None

    (input_indices, weights_arr, output_mask, output_weights), _ = jax.lax.scan(
        process_one_unit,
        (input_indices, weights_arr, output_mask, output_weights),
        jnp.arange(max_new_units),
    )
    return eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.output_mask, m.output_weights),
        model,
        (input_indices, weights_arr, output_mask, output_weights),
    )


def column_generate(
    model: DynamicNetwork,
    unit_stats: UnitStats,
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
    output_connect_strategy: str = 'all',  # kept for interface compat, ignored
    *,
    n_tasks: int,
):
    """Generate units whose connections are restricted to their task column.

    Column for slot (l, i): c = i // (max_units_per_layer // n_tasks).
    Input sources limited to column c's input range.
    Hidden sources limited to prior-layer units in column c.
    Output connections limited to column c's output range.

    Returns same tuple as random_generate:
        (model, unit_stats, new_budget, gen_mask_2d, gen_info)
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size
    output_dim = model.output_dim

    col_size = max_units // n_tasks
    input_col_size = input_dim // n_tasks

    n_total_slots = max_layers * max_units
    max_new_units = min(max_new_units, n_total_slots)

    rng, slot_rng, sample_rng, output_rng = jax.random.split(rng, 4)
    max_out = max(1, model.max_fan_out // 2)

    # --- Find and shuffle inactive slots ---
    inactive_flat = (model.unit_mask == 0).reshape(-1)  # (max_layers * max_units,)

    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_cols = cand_units // col_size          # (max_new_units,)
    cand_valid = inactive_flat[cand_flat_idx]

    # --- Build column-aware source mask: (max_layers, n_tasks, buffer_size) ---
    buf_idx = jnp.arange(buffer_size)
    n_tasks_arr = jnp.arange(n_tasks)

    # Input availability per column: (n_tasks, buffer_size)
    is_input = buf_idx < input_dim
    input_col_of_each = buf_idx // input_col_size
    input_avail_per_col = (
        is_input[None, :] & (input_col_of_each[None, :] == n_tasks_arr[:, None])
    )  # (n_tasks, buffer_size)

    # Hidden availability per column per layer: (max_layers, n_tasks, buffer_size)
    is_hidden = buf_idx >= input_dim
    safe_buf = jnp.maximum(buf_idx - input_dim, 0)
    hidden_layer_of_j = safe_buf // max_units     # (buffer_size,)
    hidden_unit_of_j = safe_buf % max_units       # (buffer_size,)
    hidden_col_of_j = hidden_unit_of_j // col_size
    hidden_is_active = model.unit_mask[hidden_layer_of_j, hidden_unit_of_j]  # (buffer_size,)

    target_l = jnp.arange(max_layers)[:, None, None]  # (max_layers, 1, 1)
    src_l = hidden_layer_of_j[None, None, :]           # (1, 1, buffer_size)
    col_c = n_tasks_arr[None, :, None]                 # (1, n_tasks, 1)
    src_c = hidden_col_of_j[None, None, :]             # (1, 1, buffer_size)

    hidden_avail = (
        is_hidden[None, None, :]
        & (src_l < target_l)
        & (src_c == col_c)
        & hidden_is_active[None, None, :]
    )  # (max_layers, n_tasks, buffer_size)

    column_available = input_avail_per_col[None, :, :] | hidden_avail  # (max_layers, n_tasks, buffer_size)

    # --- Sample input connections per candidate ---
    sample_keys = jax.random.split(sample_rng, max_new_units)

    def sample_one_unit(key, cand_layer, cand_col):
        _, key2, key3 = jax.random.split(key, 3)

        avail = column_available[cand_layer, cand_col]  # (buffer_size,)
        n_available = jnp.sum(avail)

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

        return new_indices, new_weights, n_conns, n_available

    all_indices, all_weights, all_input_costs, all_n_avail = jax.vmap(
        sample_one_unit
    )(sample_keys, cand_layers, cand_cols)

    # Don't generate units with no available column sources
    cand_valid = cand_valid & (all_n_avail > 0)

    # Cost per unit = incoming conns + outgoing connections (mixed hidden+output pool)
    all_costs = all_input_costs + max_out

    # --- Budget check ---
    costs_if_valid = jnp.where(cand_valid, all_costs.astype(jnp.float32), 0.0)
    cumulative_cost = jnp.cumsum(costs_if_valid)
    gen_mask = cand_valid & (cumulative_cost <= budget)

    # --- Apply input connections ---
    old_indices = model.input_indices[cand_layers, cand_units]
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_indices, old_indices)
    )

    new_weights_arr = model.weights.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_weights, model.weights[cand_layers, cand_units])
    )

    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 1, model.unit_mask[cand_layers, cand_units]).astype(jnp.int32)
    )

    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, model.activation_indices[cand_layers, cand_units]).astype(jnp.int32)
    )

    # --- Apply input connections and structural arrays ---
    model = eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.unit_mask, m.activation_indices),
        model,
        (
            new_input_indices, new_weights_arr, new_unit_mask, new_activation_indices,
        ),
    )

    # --- Update unit stats ---
    new_utility = unit_stats.utility.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, init_utility, unit_stats.utility[cand_layers, cand_units])
    )
    new_age = unit_stats.age.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, unit_stats.age[cand_layers, cand_units]).astype(jnp.int32)
    )
    unit_stats = UnitStats(age=new_age, utility=new_utility, accumulator=unit_stats.accumulator)

    spent = jnp.sum(jnp.where(gen_mask, all_costs.astype(jnp.float32), 0.0))
    new_budget = budget - spent

    gen_mask_2d = jnp.zeros((max_layers, max_units), dtype=jnp.bool_)
    gen_mask_2d = gen_mask_2d.at[cand_layers, cand_units].set(gen_mask)

    gen_info = (
        cand_layers, cand_units, gen_mask,
        jnp.full(max_new_units, max_out, dtype=jnp.int32),
    )

    # --- Push outgoing connections to downstream within-column targets ---
    # unit_mask is now set for new units, so column_assign_outgoing can see them
    model = column_assign_outgoing(model, gen_info, max_new_units, output_rng, n_tasks=n_tasks)

    return model, unit_stats, new_budget, gen_mask_2d, gen_info


def column_generate_relaxed(
    model: DynamicNetwork,
    unit_stats: UnitStats,
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
    output_connect_strategy: str = 'all',
    *,
    n_tasks: int,
):
    """Variant 1+2: column-restricted incoming connections, relaxed outgoing.

    Incoming connections are still limited to within-column sources (same as
    column_generate).  Outgoing connections use assign_outgoing_relaxed with
    column_priority=True, so within-column targets are preferred but the pool
    extends to all outputs and all later-layer hidden units.  The number of
    outgoing connections per unit is sampled uniformly from [1, max_out].
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size
    output_dim = model.output_dim

    col_size = max_units // n_tasks
    input_col_size = input_dim // n_tasks

    n_total_slots = max_layers * max_units
    max_new_units = min(max_new_units, n_total_slots)

    rng, slot_rng, sample_rng, output_rng, nout_rng = jax.random.split(rng, 5)
    max_out = max(1, model.max_fan_out // 2)

    # --- Find and shuffle inactive slots ---
    inactive_flat = (model.unit_mask == 0).reshape(-1)
    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_cols = cand_units // col_size
    cand_valid = inactive_flat[cand_flat_idx]

    # --- Build column-aware source mask (same as column_generate) ---
    buf_idx = jnp.arange(buffer_size)
    n_tasks_arr = jnp.arange(n_tasks)

    is_input = buf_idx < input_dim
    input_col_of_each = buf_idx // input_col_size
    input_avail_per_col = (
        is_input[None, :] & (input_col_of_each[None, :] == n_tasks_arr[:, None])
    )

    is_hidden = buf_idx >= input_dim
    safe_buf = jnp.maximum(buf_idx - input_dim, 0)
    hidden_layer_of_j = safe_buf // max_units
    hidden_unit_of_j = safe_buf % max_units
    hidden_col_of_j = hidden_unit_of_j // col_size
    hidden_is_active = model.unit_mask[hidden_layer_of_j, hidden_unit_of_j]

    target_l = jnp.arange(max_layers)[:, None, None]
    src_l = hidden_layer_of_j[None, None, :]
    col_c = n_tasks_arr[None, :, None]
    src_c = hidden_col_of_j[None, None, :]

    hidden_avail = (
        is_hidden[None, None, :]
        & (src_l < target_l)
        & (src_c == col_c)
        & hidden_is_active[None, None, :]
    )
    column_available = input_avail_per_col[None, :, :] | hidden_avail

    # --- Sample input connections per candidate ---
    sample_keys = jax.random.split(sample_rng, max_new_units)

    def sample_one_unit(key, cand_layer, cand_col):
        _, key2, key3 = jax.random.split(key, 3)
        avail = column_available[cand_layer, cand_col]
        n_available = jnp.sum(avail)
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
        return new_indices, new_weights, n_conns, n_available

    all_indices, all_weights, all_input_costs, all_n_avail = jax.vmap(
        sample_one_unit
    )(sample_keys, cand_layers, cand_cols)

    cand_valid = cand_valid & (all_n_avail > 0)

    # Random n_out per unit in [1, max_out]
    n_out_per_unit = jax.random.randint(nout_rng, (max_new_units,), 1, max_out + 1)
    all_costs = all_input_costs + n_out_per_unit

    costs_if_valid = jnp.where(cand_valid, all_costs.astype(jnp.float32), 0.0)
    cumulative_cost = jnp.cumsum(costs_if_valid)
    gen_mask = cand_valid & (cumulative_cost <= budget)

    old_indices = model.input_indices[cand_layers, cand_units]
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_indices, old_indices)
    )
    new_weights_arr = model.weights.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_weights, model.weights[cand_layers, cand_units])
    )
    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 1, model.unit_mask[cand_layers, cand_units]).astype(jnp.int32)
    )
    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, model.activation_indices[cand_layers, cand_units]).astype(jnp.int32)
    )
    model = eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.unit_mask, m.activation_indices),
        model,
        (new_input_indices, new_weights_arr, new_unit_mask, new_activation_indices),
    )

    new_utility = unit_stats.utility.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, init_utility, unit_stats.utility[cand_layers, cand_units])
    )
    new_age = unit_stats.age.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, unit_stats.age[cand_layers, cand_units]).astype(jnp.int32)
    )
    unit_stats = UnitStats(age=new_age, utility=new_utility, accumulator=unit_stats.accumulator)

    spent = jnp.sum(jnp.where(gen_mask, all_costs.astype(jnp.float32), 0.0))
    new_budget = budget - spent

    gen_mask_2d = jnp.zeros((max_layers, max_units), dtype=jnp.bool_)
    gen_mask_2d = gen_mask_2d.at[cand_layers, cand_units].set(gen_mask)

    gen_info = (cand_layers, cand_units, gen_mask, n_out_per_unit)

    model = assign_outgoing_relaxed(
        model, gen_info, max_new_units, output_rng, n_tasks=n_tasks, column_priority=True,
    )
    return model, unit_stats, new_budget, gen_mask_2d, gen_info


def free_generate(
    model: DynamicNetwork,
    unit_stats: UnitStats,
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
    output_connect_strategy: str = 'all',
    *,
    n_tasks: int,
):
    """Variant 3: no column restrictions on either incoming or outgoing connections.

    Incoming connections are sampled from any available source (all inputs and
    all prior-layer active hidden units).  Outgoing connections use
    assign_outgoing_relaxed with column_priority=False, so targets are chosen
    purely at random from all outputs and all later-layer hidden units.
    The number of outgoing connections per unit is sampled from [1, max_out].
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size

    n_total_slots = max_layers * max_units
    max_new_units = min(max_new_units, n_total_slots)

    rng, slot_rng, sample_rng, output_rng, nout_rng = jax.random.split(rng, 5)
    max_out = max(1, model.max_fan_out // 2)

    # --- Find and shuffle inactive slots ---
    inactive_flat = (model.unit_mask == 0).reshape(-1)
    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_valid = inactive_flat[cand_flat_idx]

    # --- Unrestricted source availability: all inputs + prior-layer active hidden ---
    buf_idx = jnp.arange(buffer_size)
    is_input = buf_idx < input_dim
    is_hidden = buf_idx >= input_dim
    safe_buf = jnp.maximum(buf_idx - input_dim, 0)
    hidden_layer_of_j = safe_buf // max_units
    hidden_is_active = model.unit_mask[hidden_layer_of_j, safe_buf % max_units]

    target_l = jnp.arange(max_layers)[:, None]   # (max_layers, 1)
    src_l = hidden_layer_of_j[None, :]            # (1, buffer_size)

    available = (
        is_input[None, :]
        | (is_hidden[None, :] & (src_l < target_l) & hidden_is_active[None, :])
    )  # (max_layers, buffer_size)

    # --- Sample input connections per candidate ---
    sample_keys = jax.random.split(sample_rng, max_new_units)

    def sample_one_unit(key, cand_layer):
        _, key2, key3 = jax.random.split(key, 3)
        avail = available[cand_layer]
        n_available = jnp.sum(avail)
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
        return new_indices, new_weights, n_conns, n_available

    all_indices, all_weights, all_input_costs, all_n_avail = jax.vmap(
        sample_one_unit
    )(sample_keys, cand_layers)

    cand_valid = cand_valid & (all_n_avail > 0)

    n_out_per_unit = jax.random.randint(nout_rng, (max_new_units,), 1, max_out + 1)
    all_costs = all_input_costs + n_out_per_unit

    costs_if_valid = jnp.where(cand_valid, all_costs.astype(jnp.float32), 0.0)
    cumulative_cost = jnp.cumsum(costs_if_valid)
    gen_mask = cand_valid & (cumulative_cost <= budget)

    old_indices = model.input_indices[cand_layers, cand_units]
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_indices, old_indices)
    )
    new_weights_arr = model.weights.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_weights, model.weights[cand_layers, cand_units])
    )
    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 1, model.unit_mask[cand_layers, cand_units]).astype(jnp.int32)
    )
    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, model.activation_indices[cand_layers, cand_units]).astype(jnp.int32)
    )
    model = eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.unit_mask, m.activation_indices),
        model,
        (new_input_indices, new_weights_arr, new_unit_mask, new_activation_indices),
    )

    new_utility = unit_stats.utility.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, init_utility, unit_stats.utility[cand_layers, cand_units])
    )
    new_age = unit_stats.age.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, unit_stats.age[cand_layers, cand_units]).astype(jnp.int32)
    )
    unit_stats = UnitStats(age=new_age, utility=new_utility, accumulator=unit_stats.accumulator)

    spent = jnp.sum(jnp.where(gen_mask, all_costs.astype(jnp.float32), 0.0))
    new_budget = budget - spent

    gen_mask_2d = jnp.zeros((max_layers, max_units), dtype=jnp.bool_)
    gen_mask_2d = gen_mask_2d.at[cand_layers, cand_units].set(gen_mask)

    gen_info = (cand_layers, cand_units, gen_mask, n_out_per_unit)

    model = assign_outgoing_relaxed(
        model, gen_info, max_new_units, output_rng, n_tasks=n_tasks, column_priority=False,
    )
    return model, unit_stats, new_budget, gen_mask_2d, gen_info


# ---------------------------------------------------------------------------
# Mixed generation (column + free with tag protection)
# ---------------------------------------------------------------------------

def assign_outgoing_protected(
    model: DynamicNetwork,
    gen_info: tuple,
    max_new_units: int,
    rng: PRNGKeyArray,
    *,
    n_tasks: int,
    protected_tags: Array,
) -> DynamicNetwork:
    """Like assign_outgoing_relaxed(column_priority=False) but excludes
    cross-column connections to units tagged as column-constrained.

    Free-generated units can connect to any available target EXCEPT hidden
    units that are column-tagged and in a different column from the source.
    """
    cand_layers, cand_units, gen_mask_flat, n_out_per_unit = gen_info

    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    input_dim = model.input_dim
    output_dim = model.output_dim
    max_fan_out = model.max_fan_out
    max_out = max(1, max_fan_out // 2)

    col_size = max_units // n_tasks
    pool_size = output_dim + max_layers * max_units
    keys = jax.random.split(rng, max_new_units)

    input_indices = model.input_indices
    weights_arr = model.weights
    output_mask = model.output_mask
    output_weights = model.output_weights
    unit_mask = model.unit_mask

    # Pre-compute column-tag protection mask for hidden pool
    # tag_flat: (max_layers * max_units,)
    tag_flat = protected_tags.reshape(-1)
    unit_col_arr = jnp.arange(max_units) // col_size  # (max_units,)
    hid_col = jnp.tile(unit_col_arr, max_layers)  # (max_layers * max_units,)

    def process_one_unit(carry, idx):
        input_indices, weights_arr, output_mask, output_weights = carry

        key = keys[idx]
        is_valid = gen_mask_flat[idx]
        source_layer = cand_layers[idx]
        source_unit = cand_units[idx]
        source_col = source_unit // col_size
        source_bp = input_dim + source_layer * max_units + source_unit
        n_out = n_out_per_unit[idx]

        # All outputs are available targets
        output_available = jnp.ones(output_dim, dtype=jnp.bool_)

        # Hidden targets: later layer, active, has empty slot
        has_empty_slot = jnp.any(input_indices == -1, axis=-1)
        is_later = jnp.arange(max_layers)[:, None] > source_layer
        hidden_available = (
            has_empty_slot & is_later & (unit_mask == 1)
        ).reshape(-1)

        # Exclude cross-column connections to tagged units
        protected_cross = (tag_flat == 1) & (hid_col != source_col)
        hidden_available = hidden_available & ~protected_cross

        pool_available = jnp.concatenate([output_available, hidden_available])

        noise = jax.random.uniform(key, (pool_size,))
        sort_key = jnp.where(pool_available, noise, 2.0)
        selected = jnp.argsort(sort_key)[:max_out]
        selected_valid = (jnp.arange(max_out) < n_out) & is_valid

        def assign_one_target(carry, j):
            input_indices, weights_arr, output_mask, output_weights = carry
            target_pool_idx = selected[j]
            should_assign = selected_valid[j] & pool_available[target_pool_idx]

            is_out = target_pool_idx < output_dim

            safe_out = jnp.where(is_out, target_pool_idx, 0)
            out_new = jnp.where(
                should_assign & is_out, 1, output_mask[safe_out, source_bp]
            ).astype(jnp.int32)
            output_mask = output_mask.at[safe_out, source_bp].set(out_new)
            ow_new = jnp.where(
                should_assign & is_out, 0.0, output_weights[safe_out, source_bp]
            )
            output_weights = output_weights.at[safe_out, source_bp].set(ow_new)

            hid_flat = target_pool_idx - output_dim
            tl = hid_flat // max_units
            tu = hid_flat % max_units
            safe_tl = jnp.where(~is_out, tl, 0)
            safe_tu = jnp.where(~is_out, tu, 0)

            slots = input_indices[safe_tl, safe_tu]
            first_empty = jnp.argmax(slots == -1)
            has_slot = jnp.any(slots == -1)
            should_hid = should_assign & ~is_out & has_slot

            idx_new = jnp.where(
                should_hid, source_bp, input_indices[safe_tl, safe_tu, first_empty]
            ).astype(jnp.int32)
            input_indices = input_indices.at[safe_tl, safe_tu, first_empty].set(idx_new)
            w_new = jnp.where(should_hid, 0.0, weights_arr[safe_tl, safe_tu, first_empty])
            weights_arr = weights_arr.at[safe_tl, safe_tu, first_empty].set(w_new)

            return (input_indices, weights_arr, output_mask, output_weights), None

        (input_indices, weights_arr, output_mask, output_weights), _ = jax.lax.scan(
            assign_one_target,
            (input_indices, weights_arr, output_mask, output_weights),
            jnp.arange(max_out),
        )
        return (input_indices, weights_arr, output_mask, output_weights), None

    (input_indices, weights_arr, output_mask, output_weights), _ = jax.lax.scan(
        process_one_unit,
        (input_indices, weights_arr, output_mask, output_weights),
        jnp.arange(max_new_units),
    )
    return eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.output_mask, m.output_weights),
        model,
        (input_indices, weights_arr, output_mask, output_weights),
    )


def free_generate_protected(
    model: DynamicNetwork,
    unit_stats: UnitStats,
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
    output_connect_strategy: str = 'all',
    *,
    n_tasks: int,
    column_tag: Array,
):
    """Like free_generate but outgoing connections respect column_tag protection.

    Free-generated units cannot push outgoing connections to column-tagged units
    in a different column.  Uses assign_outgoing_protected instead of
    assign_outgoing_relaxed.
    """
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    input_dim = model.input_dim
    buffer_size = model.buffer_size

    n_total_slots = max_layers * max_units
    max_new_units = min(max_new_units, n_total_slots)

    rng, slot_rng, sample_rng, output_rng, nout_rng = jax.random.split(rng, 5)
    max_out = max(1, model.max_fan_out // 2)

    # --- Find and shuffle inactive slots ---
    inactive_flat = (model.unit_mask == 0).reshape(-1)
    noise = jax.random.uniform(slot_rng, (n_total_slots,))
    sort_key = jnp.where(inactive_flat, noise, 2.0)
    perm = jnp.argsort(sort_key)
    cand_flat_idx = perm[:max_new_units]
    cand_layers = cand_flat_idx // max_units
    cand_units = cand_flat_idx % max_units
    cand_valid = inactive_flat[cand_flat_idx]

    # --- Unrestricted source availability ---
    buf_idx = jnp.arange(buffer_size)
    is_input = buf_idx < input_dim
    is_hidden = buf_idx >= input_dim
    safe_buf = jnp.maximum(buf_idx - input_dim, 0)
    hidden_layer_of_j = safe_buf // max_units
    hidden_is_active = model.unit_mask[hidden_layer_of_j, safe_buf % max_units]

    target_l = jnp.arange(max_layers)[:, None]
    src_l = hidden_layer_of_j[None, :]

    available = (
        is_input[None, :]
        | (is_hidden[None, :] & (src_l < target_l) & hidden_is_active[None, :])
    )

    sample_keys = jax.random.split(sample_rng, max_new_units)

    def sample_one_unit(key, cand_layer):
        _, key2, key3 = jax.random.split(key, 3)
        avail = available[cand_layer]
        n_available = jnp.sum(avail)
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
        return new_indices, new_weights, n_conns, n_available

    all_indices, all_weights, all_input_costs, all_n_avail = jax.vmap(
        sample_one_unit
    )(sample_keys, cand_layers)

    cand_valid = cand_valid & (all_n_avail > 0)

    n_out_per_unit = jax.random.randint(nout_rng, (max_new_units,), 1, max_out + 1)
    all_costs = all_input_costs + n_out_per_unit

    costs_if_valid = jnp.where(cand_valid, all_costs.astype(jnp.float32), 0.0)
    cumulative_cost = jnp.cumsum(costs_if_valid)
    gen_mask = cand_valid & (cumulative_cost <= budget)

    old_indices = model.input_indices[cand_layers, cand_units]
    new_input_indices = model.input_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_indices, old_indices)
    )
    new_weights_arr = model.weights.at[cand_layers, cand_units].set(
        jnp.where(gen_mask[:, None], all_weights, model.weights[cand_layers, cand_units])
    )
    new_unit_mask = model.unit_mask.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 1, model.unit_mask[cand_layers, cand_units]).astype(jnp.int32)
    )
    new_activation_indices = model.activation_indices.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, model.activation_indices[cand_layers, cand_units]).astype(jnp.int32)
    )
    model = eqx.tree_at(
        lambda m: (m.input_indices, m.weights, m.unit_mask, m.activation_indices),
        model,
        (new_input_indices, new_weights_arr, new_unit_mask, new_activation_indices),
    )

    new_utility = unit_stats.utility.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, init_utility, unit_stats.utility[cand_layers, cand_units])
    )
    new_age = unit_stats.age.at[cand_layers, cand_units].set(
        jnp.where(gen_mask, 0, unit_stats.age[cand_layers, cand_units]).astype(jnp.int32)
    )
    unit_stats = UnitStats(age=new_age, utility=new_utility, accumulator=unit_stats.accumulator)

    spent = jnp.sum(jnp.where(gen_mask, all_costs.astype(jnp.float32), 0.0))
    new_budget = budget - spent

    gen_mask_2d = jnp.zeros((max_layers, max_units), dtype=jnp.bool_)
    gen_mask_2d = gen_mask_2d.at[cand_layers, cand_units].set(gen_mask)

    gen_info = (cand_layers, cand_units, gen_mask, n_out_per_unit)

    model = assign_outgoing_protected(
        model, gen_info, max_new_units, output_rng,
        n_tasks=n_tasks, protected_tags=column_tag,
    )
    return model, unit_stats, new_budget, gen_mask_2d, gen_info


def mixed_generate(
    model: DynamicNetwork,
    unit_stats: UnitStats,
    budget: Float[Array, ''],
    max_new_units: int,
    init_utility: Float[Array, ''],
    rng: PRNGKeyArray,
    output_connect_strategy: str = 'all',
    *,
    n_tasks: int,
    column_tag: Array,
):
    """Generate units: half column-constrained, half free (with tag protection).

    Column-constrained units use column_generate (within-column incoming and
    outgoing).  Free units use free_generate_protected (unrestricted incoming,
    outgoing protected from cross-column connections to tagged units).

    Returns updated column_tag as the last element of gen_info so the caller
    can extract and store it.
    """
    half_budget = budget / 2
    half_units = max_new_units // 2
    rng1, rng2 = jax.random.split(rng)

    # Column-constrained half
    model, unit_stats, remaining1, gen_mask_col, _ = column_generate(
        model, unit_stats, half_budget, half_units, init_utility, rng1,
        n_tasks=n_tasks,
    )

    # Tag newly created column units as 1
    column_tag = jnp.where(gen_mask_col, 1, column_tag).astype(jnp.int32)

    # Free half with protection — strict half budget, no leftover from column
    model, unit_stats, remaining2, gen_mask_free, gen_info = free_generate_protected(
        model, unit_stats, half_budget, half_units, init_utility, rng2,
        n_tasks=n_tasks, column_tag=column_tag,
    )

    # Tag newly created free units as 2
    column_tag = jnp.where(gen_mask_free, 2, column_tag).astype(jnp.int32)

    gen_mask_2d = gen_mask_col | gen_mask_free
    # Stash updated column_tag in gen_info for MixedConnectivityManager to extract
    gen_info_with_tag = gen_info + (column_tag,)
    total_remaining = remaining1 + remaining2
    return model, unit_stats, total_remaining, gen_mask_2d, gen_info_with_tag


class MixedConnectivityManager(ConnectivityManager):
    """ConnectivityManager extended with per-unit column tagging.

    Tracks which units were generated as column-constrained (column_tag=1) vs
    free (column_tag=0).  Overrides modify_structure to pass the tag to
    mixed_generate and extract the updated tag from gen_info.
    """
    column_tag: Array  # (max_layers, max_units_per_layer), int32

    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.column_tag = jnp.zeros(
            (model.max_layers, model.max_units_per_layer), dtype=jnp.int32)

    def modify_structure(self, model, optimizer, *, rng):
        rng, prune_rng, gen_rng = jax.random.split(rng, 3)

        # Compute initial utility for new units
        init_utility = self.utility_init_fn(
            self.unit_stats.utility, model.unit_mask,
        )

        # Prune mask
        prune_mask, n_pruned = self._make_prune_mask(model, rng=prune_rng)

        # Count freed connections before clearing
        pruned_conn_valid = (model.input_indices >= 0) & prune_mask[:, :, None]
        n_freed_incoming = jnp.sum(pruned_conn_valid).astype(jnp.float32)

        buf_positions = _unit_buf_positions(model)
        pruned_buf_mask = _prune_mask_to_buf_mask(
            prune_mask, buf_positions, model.buffer_size)
        n_freed_output = jnp.sum(
            model.output_mask * pruned_buf_mask[None, :].astype(jnp.int32)
        ).astype(jnp.float32)
        n_freed = n_freed_incoming + n_freed_output

        # Batch prune
        prune_3d = prune_mask[:, :, None]
        new_unit_mask = jnp.where(prune_mask, 0, model.unit_mask).astype(jnp.int32)
        new_weights = jnp.where(prune_3d, 0.0, model.weights)
        new_input_indices = jnp.where(
            prune_3d, -1, model.input_indices).astype(jnp.int32)

        pruned_buf_2d = pruned_buf_mask[None, :]
        new_output_mask = model.output_mask * (1 - pruned_buf_2d.astype(jnp.int32))
        new_output_weights = (
            model.output_weights * (1 - pruned_buf_2d.astype(jnp.float32)))

        model = eqx.tree_at(
            lambda m: (m.unit_mask, m.weights, m.input_indices,
                       m.output_mask, m.output_weights),
            model,
            (new_unit_mask, new_weights, new_input_indices,
             new_output_mask, new_output_weights),
        )

        # Reset stats and tags for pruned units
        new_age = jnp.where(prune_mask, 0, self.unit_stats.age).astype(jnp.int32)
        new_utility = jnp.where(prune_mask, 0.0, self.unit_stats.utility)
        new_column_tag = jnp.where(
            prune_mask, 0, self.column_tag).astype(jnp.int32)

        # Generation budget
        active_conns_after_prune = (
            jnp.sum(model.input_indices >= 0)
            + jnp.sum(model.output_mask)
        ).astype(jnp.float32)
        gen_budget = jnp.maximum(
            0.0, self.connection_budget - active_conns_after_prune)

        new_accumulator = jnp.maximum(0.0, self.unit_stats.accumulator - n_freed)
        unit_stats = UnitStats(
            age=new_age, utility=new_utility, accumulator=new_accumulator)

        # Generate new units — pass column_tag via kwarg
        model, unit_stats, _, gen_mask_2d, gen_info = self.generate_fn(
            model, unit_stats, gen_budget,
            self.max_new_units_per_step,
            init_utility, gen_rng,
            output_connect_strategy=self.output_connect_strategy,
            column_tag=new_column_tag,
        )

        # Extract updated column_tag from gen_info (last element)
        updated_tag = gen_info[-1]

        # Assign sparse outgoing if needed
        if self.output_connect_strategy == 'random_sparse':
            rng, sparse_rng = jax.random.split(rng)
            model = assign_sparse_outgoing(
                model, gen_info[:-1], self.max_new_units_per_step, sparse_rng,
                output_weight_init=self.output_weight_init,
            )

        # Rebuild outgoing indices
        model = build_outgoing_indices(model)

        # Reset optimizer state
        affected_mask = prune_mask | gen_mask_2d
        optimizer = _reset_optimizer_state(optimizer, model, affected_mask)

        # Per-layer counts
        pruned_per_layer = prune_mask.sum(axis=1).astype(jnp.int32)
        generated_per_layer = gen_mask_2d.sum(axis=1).astype(jnp.int32)

        new_manager = tree_replace(
            self, unit_stats=unit_stats, rng=rng, column_tag=updated_tag)
        return new_manager, model, optimizer, pruned_per_layer, generated_per_layer


class ConnectionMixedConnectivityManager(ConnectionConnectivityManager):
    """ConnectionConnectivityManager with per-unit column tagging.

    Prunes individual connections (connection-level), but tracks column_tag
    for mixed_generate and passes it through the generate cycle, mirroring
    MixedConnectivityManager's tag logic.
    """
    column_tag: Array  # (max_layers, max_units_per_layer), int32

    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.column_tag = jnp.zeros(
            (model.max_layers, model.max_units_per_layer), dtype=jnp.int32)

    def modify_structure(self, model, optimizer, *, rng):
        rng, prune_rng, gen_rng = jax.random.split(rng, 3)
        max_layers = model.max_layers
        max_units = model.max_units_per_layer

        # ===== Phase 1: Global connection pruning =====
        hidden_prune_3d, output_prune_2d, n_pruned = (
            self._make_connection_prune_masks(model, rng=prune_rng))

        new_weights = jnp.where(hidden_prune_3d, 0.0, model.weights)
        new_input_indices = jnp.where(
            hidden_prune_3d, -1, model.input_indices).astype(jnp.int32)
        new_output_weights = jnp.where(output_prune_2d, 0.0, model.output_weights)
        new_output_mask = jnp.where(
            output_prune_2d, 0, model.output_mask).astype(jnp.int32)

        new_hidden_utility = jnp.where(
            hidden_prune_3d, 0.0, self.connection_stats.hidden_utility)
        new_output_utility = jnp.where(
            output_prune_2d, 0.0, self.connection_stats.output_utility)

        # ===== Phase 2: Dead unit detection (0 outgoing) =====
        buf_positions = _unit_buf_positions(model)

        active_mask = (new_input_indices >= 0)
        flat_idx = jnp.where(active_mask, new_input_indices, 0).reshape(-1)
        flat_active = active_mask.reshape(-1).astype(jnp.int32)
        h2h_counts = jnp.zeros(model.buffer_size, dtype=jnp.int32)
        h2h_counts = h2h_counts.at[flat_idx].add(flat_active)
        h2o_counts = new_output_mask.sum(axis=0)

        total_outgoing = h2h_counts + h2o_counts
        unit_outgoing = total_outgoing[buf_positions]
        dead_mask = (model.unit_mask == 1) & (unit_outgoing == 0)

        dead_3d = dead_mask[:, :, None]
        new_unit_mask = jnp.where(dead_mask, 0, model.unit_mask).astype(jnp.int32)
        new_weights = jnp.where(dead_3d, 0.0, new_weights)
        new_input_indices = jnp.where(dead_3d, -1, new_input_indices).astype(jnp.int32)

        dead_buf_mask = _prune_mask_to_buf_mask(
            dead_mask, buf_positions, model.buffer_size)
        new_output_mask = new_output_mask * (1 - dead_buf_mask[None, :].astype(jnp.int32))
        new_output_weights = new_output_weights * (1 - dead_buf_mask[None, :].astype(jnp.float32))

        new_hidden_utility = jnp.where(dead_3d, 0.0, new_hidden_utility)
        new_output_utility = new_output_utility * (
            1 - dead_buf_mask[None, :].astype(jnp.float32))

        # Reset column_tag for dead units
        new_column_tag = jnp.where(
            dead_mask, 0, self.column_tag).astype(jnp.int32)

        model = eqx.tree_at(
            lambda m: (m.unit_mask, m.weights, m.input_indices,
                       m.output_mask, m.output_weights),
            model,
            (new_unit_mask, new_weights, new_input_indices,
             new_output_mask, new_output_weights),
        )

        # ===== Phase 3: Generate new units =====
        active_conns_after = (
            jnp.sum(model.input_indices >= 0) + jnp.sum(model.output_mask)
        ).astype(jnp.float32)
        gen_budget = jnp.maximum(0.0, self.connection_budget - active_conns_after)

        unit_shape = (max_layers, max_units)
        temp_unit_stats = UnitStats(
            age=jnp.zeros(unit_shape, dtype=jnp.int32),
            utility=jnp.zeros(unit_shape, dtype=jnp.float32),
            accumulator=jnp.array(0.0),
        )

        model, _temp_stats, _, gen_mask_2d, gen_info = self.generate_fn(
            model, temp_unit_stats, gen_budget,
            self.max_new_units_per_step,
            jnp.array(0.0),
            gen_rng,
            output_connect_strategy=self.output_connect_strategy,
            column_tag=new_column_tag,
        )

        # Extract updated column_tag from gen_info (last element)
        updated_tag = gen_info[-1]

        # ===== Phase 4: Finalize =====
        if self.output_connect_strategy == 'random_sparse':
            rng, sparse_rng = jax.random.split(rng)
            model = assign_sparse_outgoing(
                model, gen_info[:-1], self.max_new_units_per_step, sparse_rng,
                output_weight_init=self.output_weight_init,
            )

        model = build_outgoing_indices(model)

        optimizer = _reset_optimizer_state_connections(
            optimizer, model,
            hidden_prune_3d, output_prune_2d,
            dead_mask, gen_mask_2d,
        )

        new_acc = jnp.maximum(
            0.0, self.connection_stats.accumulator - n_pruned.astype(jnp.float32))

        new_stats = ConnectionStats(
            hidden_utility=new_hidden_utility,
            output_utility=new_output_utility,
            accumulator=new_acc,
        )

        affected = dead_mask | gen_mask_2d
        new_unit_utility = jnp.where(affected, 0.0, self.unit_stats.utility)
        new_unit_age = jnp.where(affected, 0, self.unit_stats.age).astype(jnp.int32)
        new_unit_stats = UnitStats(
            age=new_unit_age, utility=new_unit_utility, accumulator=new_acc)

        pruned_per_layer = dead_mask.sum(axis=1).astype(jnp.int32)
        generated_per_layer = gen_mask_2d.sum(axis=1).astype(jnp.int32)

        new_manager = tree_replace(
            self, connection_stats=new_stats, unit_stats=new_unit_stats,
            rng=rng, column_tag=updated_tag,
        )
        return new_manager, model, optimizer, pruned_per_layer, generated_per_layer


# ---------------------------------------------------------------------------
# Column-constrained initialization
# ---------------------------------------------------------------------------

def _column_init(
    model: DynamicNetwork,
    n_tasks: int,
    key: PRNGKeyArray,
) -> DynamicNetwork:
    """Rewire an existing network so units are evenly distributed across columns
    and all connections are within-column.

    Takes a model that was already created by ``init_random_dynamic_network``
    (with random cross-column connections) and replaces its connectivity with
    purely within-column wiring.  Units are distributed as ``hidden_dim //
    n_tasks`` per column per layer (e.g. 104 with hidden_dim=520, n_tasks=5).

    The model's static fields (activation_fns, etc.) are preserved so that it
    remains stackable with models from the same constructor call.
    """
    input_dim = model.input_dim
    output_dim = model.output_dim
    max_layers = model.max_layers
    max_units = model.max_units_per_layer
    max_conns = model.max_connections_per_unit
    buffer_size = model.buffer_size

    col_size = max_units // n_tasks
    input_col_size = input_dim // n_tasks
    out_col_size = output_dim // n_tasks

    # Count current active units to determine units_per_col
    # (uses the first layer's active count as reference)
    hidden_dim = int(model.unit_mask[0].sum())
    units_per_col = hidden_dim // n_tasks

    # Start from zeroed arrays
    weights = jnp.zeros_like(model.weights)
    input_indices = jnp.full_like(model.input_indices, -1)
    unit_mask = jnp.zeros_like(model.unit_mask)
    output_mask = jnp.zeros_like(model.output_mask)
    output_weights = jnp.zeros_like(model.output_weights)

    for c in range(n_tasks):
        col_start = c * col_size
        col_end = col_start + units_per_col
        inp_start = c * input_col_size
        inp_end = inp_start + input_col_size

        for l in range(max_layers):
            avail_positions = list(range(inp_start, inp_end))
            for prev_l in range(l):
                offset = input_dim + prev_l * max_units
                avail_positions.extend(range(offset + col_start, offset + col_end))
            avail_arr = jnp.array(avail_positions, dtype=jnp.int32)
            n_available = len(avail_positions)
            half_conns = max(1, max_conns // 2)
            n_conns = min(n_available, half_conns)

            key, layer_key = jax.random.split(key)
            unit_keys = jax.random.split(layer_key, units_per_col)

            def _sample_unit(unit_key, avail=avail_arr, n_avail=n_available,
                             n_c=n_conns):
                perm = jax.random.permutation(unit_key, n_avail)[:n_c]
                sources = avail[perm]
                padded = jnp.full(max_conns, -1, dtype=jnp.int32)
                padded = padded.at[:n_c].set(sources)
                return padded

            layer_indices = jax.vmap(_sample_unit)(unit_keys)

            key, w_key = jax.random.split(key)
            bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.float32(max(n_conns, 1)))
            w = jax.random.uniform(
                w_key, (units_per_col, max_conns), minval=-bound, maxval=bound)
            conn_mask = (layer_indices >= 0).astype(jnp.float32)
            w = w * conn_mask

            input_indices = input_indices.at[l, col_start:col_end].set(layer_indices)
            weights = weights.at[l, col_start:col_end].set(w)
            unit_mask = unit_mask.at[l, col_start:col_end].set(1)

        # Output connections: last-layer units → same-column outputs
        last_offset = input_dim + (max_layers - 1) * max_units
        buf_positions = jnp.arange(last_offset + col_start, last_offset + col_end)
        out_indices = jnp.arange(c * out_col_size, (c + 1) * out_col_size)
        output_mask = output_mask.at[
            out_indices[:, None], buf_positions[None, :]
        ].set(1)
        key, ow_key = jax.random.split(key)
        ow = lecun_uniform(ow_key, (out_col_size, units_per_col),
                           in_dim=units_per_col)
        output_weights = output_weights.at[
            out_indices[:, None], buf_positions[None, :]
        ].set(ow)

    model = eqx.tree_at(
        lambda n: (n.weights, n.input_indices, n.unit_mask,
                   n.output_mask, n.output_weights),
        model,
        (weights, input_indices, unit_mask, output_mask, output_weights),
    )
    return build_outgoing_indices(model)


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: DynamicNetwork
    optimizer: eqx.Module
    structure_tracker: ConnectivityManager
    step: jax.Array
    rng: PRNGKeyArray


def _make_filter_spec(model: DynamicNetwork):
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(lambda n: (n.weights, n.output_weights), spec, (True, True))


def _sparsify_outputs(model: DynamicNetwork, rng: PRNGKeyArray) -> DynamicNetwork:
    """Replace dense all-to-all output connections with sparse random connections.

    At initialization, init_random_dynamic_network connects every last-layer unit
    to all output_dim outputs.  This replaces those dense connections with
    max_fan_out // 2 randomly selected outputs per unit — matching the number
    that column_generate uses when producing new units.
    """
    n_keep = model.max_fan_out // 2
    last_layer = model.max_layers - 1
    max_units = model.max_units_per_layer
    input_dim = model.input_dim
    output_dim = model.output_dim

    buf_positions = input_dim + last_layer * max_units + jnp.arange(max_units)
    keys = jax.random.split(rng, max_units)

    def sparsify_unit(key):
        noise = jax.random.uniform(key, (output_dim,))
        keep = jnp.argsort(noise)[:n_keep]
        return jnp.zeros(output_dim, dtype=jnp.int32).at[keep].set(1)

    new_masks = jax.vmap(sparsify_unit)(keys)  # (max_units, output_dim)

    # Zero out inactive units so they don't accidentally get output connections
    active = model.unit_mask[last_layer]  # (max_units,)
    new_masks = new_masks * active[:, None]

    new_om = model.output_mask.at[:, buf_positions].set(new_masks.T)
    new_ow = jnp.where(new_om > 0, model.output_weights, 0.0)
    return eqx.tree_at(
        lambda m: (m.output_mask, m.output_weights),
        model,
        (new_om, new_ow),
    )


def prepare_experiment(cfg: DictConfig, n_tasks: int):
    seeds = cfg.seed
    base_images, base_labels, num_classes, input_dim_per_task = load_dataset('mnist', split='train')
    input_dim = n_tasks * input_dim_per_task
    output_dim = n_tasks * num_classes

    test_images_raw = test_labels_raw = None
    if cfg.train.get('eval_freq', 0) > 0:
        test_images_raw, test_labels_raw, _, _ = load_dataset('mnist', split='test')

    streams, train_states = [], []
    for seed in seeds:
        rng = jax.random.key(seed)
        model_key = rng_from_string(rng, 'model')
        output_init_key = rng_from_string(rng, 'output_init')

        streams.append(ParallelMNISTStream(
            images=base_images, labels=base_labels,
            n_tasks=n_tasks, batch_size=cfg.train.batch_size,
            seed=seed, permute_period=cfg.dataset.get('permute_period', 0),
            permute_stop=cfg.dataset.get('permute_stop', 0),
            test_images=test_images_raw, test_labels=test_labels_raw,
        ))

        model = init_random_dynamic_network(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=cfg.model.n_layers,
            units_per_layer=cfg.model.hidden_dim,
            max_units_per_layer=cfg.model.get('max_units_per_layer', None),
            max_connections_per_unit=cfg.model.get('max_connections_per_unit', None),
            activations=(cfg.model.activation,),
            max_fan_out=cfg.model.get('max_fan_out', None),
            connect_all_to_output=cfg.model.get('connect_all_to_output', False),
            init_strategy=cfg.model.get('init_strategy', 'linear'),
            key=model_key,
        )
        # Replace dense all-to-all output init with sparse random connections
        model = _sparsify_outputs(model, output_init_key)

        optimizer = prepare_optimizer(
            model, cfg.optimizer.name, cfg.optimizer,
            filter_spec=_make_filter_spec(model),
        )

        variant = cfg.get('variant', 'column_guided')
        if variant == 'relaxed_outputs':
            utility_fn = partial(column_utility, n_tasks=n_tasks)
            generate_fn = partial(column_generate_relaxed, n_tasks=n_tasks)
        elif variant == 'normal_utility':
            utility_fn = normalized_contribution_utility
            generate_fn = partial(column_generate_relaxed, n_tasks=n_tasks)
        elif variant == 'no_column':
            utility_fn = normalized_contribution_utility
            generate_fn = partial(free_generate, n_tasks=n_tasks)
        elif variant == 'utility_comparison':
            utility_fn = normalized_contribution_utility
            generate_fn = partial(column_generate, n_tasks=n_tasks)  # unused
        elif variant == 'mixed_generation':
            utility_fn = normalized_contribution_utility
            generate_fn = partial(mixed_generate, n_tasks=n_tasks)
        else:  # 'column_guided' (original)
            utility_fn = partial(column_utility, n_tasks=n_tasks)
            generate_fn = partial(column_generate, n_tasks=n_tasks)

        # Column-constrained initialization (works with any variant)
        if cfg.get('init_mode', 'random') == 'column':
            model = _column_init(
                model, n_tasks=n_tasks,
                key=rng_from_string(rng, 'column_init'),
            )

        tracker_mode = cfg.structure_tracker.get('mode', 'unit')
        if tracker_mode == 'connection':
            conn_cls = (ConnectionMixedConnectivityManager
                        if variant == 'mixed_generation'
                        else ConnectionConnectivityManager)
            tracker = conn_cls(
                model=model,
                prune_rate=cfg.structure_tracker.prune_rate,
                connection_budget=cfg.structure_tracker.connection_budget,
                decay_rate=cfg.structure_tracker.decay_rate,
                max_new_units_per_step=cfg.structure_tracker.get('max_new_units_per_step', 512),
                output_connect_strategy='all',
                output_weight_init='zero',
                generate_fn=generate_fn,
                rng=rng_from_string(rng, 'tracker'),
            )
        else:
            tracker_cls = (MixedConnectivityManager
                           if variant == 'mixed_generation'
                           else ConnectivityManager)
            tracker = tracker_cls(
                model=model,
                prune_rate=cfg.structure_tracker.prune_rate,
                connection_budget=cfg.structure_tracker.connection_budget,
                decay_rate=cfg.structure_tracker.decay_rate,
                maturity_threshold=cfg.structure_tracker.maturity_threshold,
                max_new_units_per_step=cfg.structure_tracker.get('max_new_units_per_step', 512),
                output_connect_strategy='all',
                output_weight_init='zero',
                utility_fn=utility_fn,
                generate_fn=generate_fn,
                utility_init_fn=median_utility_init,
                rng=rng_from_string(rng, 'tracker'),
            )

        train_states.append(TrainState(
            model=model, optimizer=optimizer, structure_tracker=tracker,
            step=jnp.array(0), rng=rng_from_string(rng, 'train'),
        ))

    n_params = count_params(train_states[0].model)
    net = train_states[0].model
    n_units = count_active_units(net)
    n_conns = count_active_connections(net)
    max_conns = net.weights.size + net.output_weights.size
    print(f'Model: DynamicNetwork  params={n_params}  units={n_units}  '
          f'conns={n_conns}/{max_conns}  seeds={seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, streams, n_params, num_classes


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _cross_entropy_loss(logits, one_hot):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def train_step(
    train_state: TrainState,
    data,
    do_restructure: bool = False,
    num_classes: int = 10,
    n_tasks: int = 1,
) -> Tuple['TrainState', StepMetrics]:
    images, labels = data
    # labels: (batch_size, n_tasks)
    one_hot = jax.nn.one_hot(labels, num_classes)  # (batch_size, n_tasks, num_classes)

    def loss_fn(model):
        raw_outputs, param_inputs = jax.vmap(model)(images)  # (batch_size, n_tasks*num_classes)
        outputs = raw_outputs.reshape(-1, n_tasks, num_classes)
        loss = _cross_entropy_loss(outputs, one_hot)
        return loss, (raw_outputs, param_inputs)

    (loss, (raw_outputs, param_inputs)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(train_state.model)

    outputs_r = raw_outputs.reshape(-1, n_tasks, num_classes)
    predicted = jnp.argmax(outputs_r, axis=-1)  # (batch_size, n_tasks)
    correct = (predicted == labels).astype(jnp.float32).mean()

    updates, new_optimizer = train_state.optimizer.with_update(grads, train_state.model)
    new_model = eqx.apply_updates(train_state.model, updates)
    new_model = sync_outgoing_weights(new_model)

    new_tracker = train_state.structure_tracker.update_stats(
        new_model, param_inputs, grads=grads, updates=updates,
        targets=one_hot.reshape(raw_outputs.shape), predictions=raw_outputs,
    )

    n_model_layers = new_model.max_layers
    pruned_per_layer = jnp.zeros(n_model_layers, dtype=jnp.int32)
    generated_per_layer = jnp.zeros(n_model_layers, dtype=jnp.int32)
    if do_restructure:
        rng, restructure_rng = jax.random.split(train_state.rng)
        new_tracker, new_model, new_optimizer, pruned_per_layer, generated_per_layer = (
            new_tracker.modify_structure(new_model, new_optimizer, rng=restructure_rng)
        )

    new_state = tree_replace(
        train_state,
        model=new_model,
        optimizer=new_optimizer,
        structure_tracker=new_tracker,
        step=train_state.step + 1,
    )
    return new_state, StepMetrics(
        loss=loss, correct=correct,
        pruned_per_layer=pruned_per_layer,
        generated_per_layer=generated_per_layer,
    )


# ---------------------------------------------------------------------------
# Column structure metrics
# ---------------------------------------------------------------------------

def compute_column_metrics(train_state, n_tasks: int) -> dict:
    """Compute average cross-column connections per active unit, per layer.

    Works with both single-seed and vmapped (multi-seed) TrainState.
    A connection is cross-column if its source/destination column differs from
    the unit's column (determined by slot position within the layer).

    Returns keys: layer_{l}/avg_cross_col_conns
    """
    model = train_state.model
    unit_mask = np.array(model.unit_mask)          # (..., n_layers, max_units)
    input_indices = np.array(model.input_indices)  # (..., n_layers, max_units, max_conns)
    output_mask = np.array(model.output_mask)      # (..., output_dim, buffer_size)

    multi_seed = unit_mask.ndim == 3
    if not multi_seed:
        unit_mask = unit_mask[None]
        input_indices = input_indices[None]
        output_mask = output_mask[None]

    n_seeds = unit_mask.shape[0]
    n_layers = model.max_layers
    max_units = model.max_units_per_layer
    input_dim = model.input_dim
    output_dim = model.output_dim

    col_size = max_units // n_tasks
    input_col_size = input_dim // n_tasks
    out_col_size = output_dim // n_tasks

    unit_cols = np.arange(max_units) // col_size  # (max_units,)

    metrics = {}
    for l in range(n_layers):
        cross_per_seed = []
        for s in range(n_seeds):
            active_mask = unit_mask[s, l].astype(bool)  # (max_units,)
            if not active_mask.any():
                cross_per_seed.append(0.0)
                continue

            # --- Incoming cross-column connections ---
            idx = input_indices[s, l]              # (max_units, max_conns)
            active_conn = idx >= 0

            is_input_src = active_conn & (idx < input_dim)
            is_hidden_src = active_conn & (idx >= input_dim)

            safe_idx = np.maximum(idx, 0)
            src_col = np.where(
                is_input_src,
                safe_idx // input_col_size,
                (safe_idx - input_dim) % max_units // col_size,
            )  # (max_units, max_conns)

            in_cross = active_conn & (src_col != unit_cols[:, None])  # (max_units, max_conns)
            in_cross_count = in_cross.sum(axis=1)  # (max_units,)

            # --- Outgoing cross-column output connections ---
            buf_offset = input_dim + l * max_units
            buf_positions = np.arange(buf_offset, buf_offset + max_units)
            # output_mask[s] has shape (output_dim, buffer_size); slice seed first to avoid
            # numpy advanced-index axis reordering when mixing scalar and array indexing
            out_mask_at_units = output_mask[s][:, buf_positions]  # (output_dim, max_units)
            out_cols = np.arange(output_dim) // out_col_size  # (output_dim,)
            out_cross = out_mask_at_units.astype(bool) & (out_cols[:, None] != unit_cols[None, :])
            out_cross_count = out_cross.sum(axis=0)  # (max_units,)

            total_cross = (in_cross_count + out_cross_count)[active_mask]
            cross_per_seed.append(float(total_cross.mean()))

        metrics[f'layer_{l}/avg_cross_col_conns'] = float(np.mean(cross_per_seed))

    return metrics


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def _eval_forward(model, images, labels, num_classes, n_tasks):
    outputs, _ = jax.vmap(model)(images)
    one_hot = jax.nn.one_hot(labels, num_classes)
    outputs_r = outputs.reshape(-1, n_tasks, num_classes)
    loss = _cross_entropy_loss(outputs_r, one_hot)
    predicted = jnp.argmax(outputs_r, axis=-1)
    correct = (predicted == labels).astype(jnp.float32).mean()
    return loss, correct


def evaluate_test(batched_model, test_images, test_labels, num_classes, n_tasks, batch_size=512):
    """Evaluate a vmapped (seed-leading) model on the full test set in chunks."""
    @jax.jit
    def _eval_chunk(model, imgs, lbls):
        return jax.vmap(
            lambda m: _eval_forward(m, imgs, lbls, num_classes, n_tasks)
        )(model)

    n_test = test_images.shape[0]
    total_loss = total_acc = None
    n_chunks = 0
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        chunk_loss, chunk_acc = _eval_chunk(
            batched_model,
            jnp.array(test_images[start:end]),
            jnp.array(test_labels[start:end]),
        )
        if total_loss is None:
            total_loss, total_acc = chunk_loss, chunk_acc
        else:
            total_loss, total_acc = total_loss + chunk_loss, total_acc + chunk_acc
        n_chunks += 1
    return total_loss / n_chunks, total_acc / n_chunks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _compute_tag_metrics(train_state, n_tasks):
    """Compute tagged vs untagged utility metrics for mixed_generation logging."""
    tracker = train_state.structure_tracker
    utility = np.array(tracker.unit_stats.utility)    # (n_seeds, L, U)
    unit_mask = np.array(train_state.model.unit_mask)  # (n_seeds, L, U)
    column_tag = np.array(tracker.column_tag)          # (n_seeds, L, U)

    active = unit_mask.astype(bool)
    tagged = active & (column_tag == 1)     # column-constrained generated
    free = active & (column_tag == 2)       # free generated
    initial = active & (column_tag == 0)    # from initialization

    metrics = {}
    for name, mask in [('tagged', tagged), ('free', free), ('initial', initial)]:
        vals = utility[mask]
        if vals.size > 0:
            metrics[f'{name}/mean_utility'] = float(vals.mean())
            metrics[f'{name}/median_utility'] = float(np.median(vals))
            metrics[f'{name}/count'] = float(mask.sum() / unit_mask.shape[0])  # avg per seed
        else:
            metrics[f'{name}/mean_utility'] = 0.0
            metrics[f'{name}/median_utility'] = 0.0
            metrics[f'{name}/count'] = 0.0

    # Pruning threshold estimate
    accumulator = np.array(tracker.unit_stats.accumulator)  # (n_seeds,)
    output_dim = train_state.model.output_dim
    thresholds = []
    for s in range(utility.shape[0]):
        active_util = utility[s][active[s]]
        if active_util.size == 0:
            thresholds.append(0.0)
            continue
        n_to_prune = int(accumulator[s] / (1 + output_dim))
        n_to_prune = min(n_to_prune, active_util.size)
        if n_to_prune <= 0:
            thresholds.append(float(active_util.min()))
        else:
            sorted_util = np.sort(active_util)
            thresholds.append(float(sorted_util[n_to_prune]))
    metrics['prune_threshold'] = float(np.mean(thresholds))

    return metrics


def _snapshot_from_state(state):
    """Extract per-unit snapshot arrays from a TrainState (inside JIT)."""
    tracker = state.structure_tracker
    tag = (tracker.column_tag
           if hasattr(tracker, 'column_tag')
           else jnp.zeros_like(state.model.unit_mask))
    return (
        tracker.unit_stats.utility,   # (max_layers, max_units)
        state.model.unit_mask,        # (max_layers, max_units)
        tracker.unit_stats.age,       # (max_layers, max_units)
        tag,                          # (max_layers, max_units)
    )


def _subsample_snapshots(snap_tuple, step_base, log_freq, subsample):
    """Subsample snapshot arrays while preserving lifecycle events.

    Detects both unit_mask transitions (0→1, 1→0) and age resets (age drops
    to 0 while mask stays 1), which indicate prune+regenerate in the same step.

    Returns (subsampled_tuple, step_indices).
    """
    # snap_tuple: tuple of 4 numpy arrays, each (n_seeds, log_freq, L, U)
    unit_mask_full = snap_tuple[1]  # (n_seeds, T, L, U)
    age_full = snap_tuple[2]        # (n_seeds, T, L, U)
    n_steps = unit_mask_full.shape[1]

    # Detect mask transitions
    mask_diff = np.diff(unit_mask_full, axis=1)
    mask_events = np.any(mask_diff != 0, axis=(0, 2, 3))  # (T-1,)

    # Detect age resets while staying active (prune+regen same step)
    active_both = (unit_mask_full[:, :-1] == 1) & (unit_mask_full[:, 1:] == 1)
    age_reset = active_both & (age_full[:, 1:] == 0) & (age_full[:, :-1] > 0)
    age_events = np.any(age_reset, axis=(0, 2, 3))  # (T-1,)

    event_steps = mask_events | age_events

    # Keep subsampled steps + steps adjacent to lifecycle events
    keep = np.zeros(n_steps, dtype=bool)
    keep[::subsample] = True
    event_indices = np.where(event_steps)[0]
    for ei in event_indices:
        keep[ei] = True
        if ei + 1 < n_steps:
            keep[ei + 1] = True

    subsampled = tuple(a[:, keep] for a in snap_tuple)
    step_indices = step_base + np.where(keep)[0]
    return subsampled, step_indices


def run_experiment(cfg, train_state, streams, num_classes, n_tasks, test_data=None):
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq
    prune_frequency = cfg.structure_tracker.get('prune_frequency', log_freq)
    eval_freq = cfg.train.get('eval_freq', 0)
    variant = cfg.get('variant', 'column_guided')
    structure_enabled = cfg.structure_tracker.get('enabled', True)
    collect_snapshots = (variant == 'mixed_generation')
    snapshot_subsample = cfg.get('snapshot_subsample', 10)

    assert log_freq % prune_frequency == 0, (
        f'log_freq ({log_freq}) must be divisible by prune_frequency ({prune_frequency})'
    )
    n_inner_blocks = log_freq // prune_frequency

    def _normal_step(state, data):
        return train_step(state, data, do_restructure=False,
                          num_classes=num_classes, n_tasks=n_tasks)

    def _normal_step_with_snapshot(state, data):
        state, metrics = train_step(state, data, do_restructure=False,
                                    num_classes=num_classes, n_tasks=n_tasks)
        return state, (metrics, _snapshot_from_state(state))

    if structure_enabled:
        if collect_snapshots:
            # Scan with restructure + per-step snapshots
            def _inner_step(state, data_block):
                normal_data = (data_block[0][:-1], data_block[1][:-1])
                state, (normal_metrics, normal_snaps) = jax.lax.scan(
                    _normal_step_with_snapshot, state, normal_data,
                    unroll=SCAN_UNROLL)
                state, restructure_metrics = train_step(
                    state, (data_block[0][-1], data_block[1][-1]),
                    do_restructure=True, num_classes=num_classes, n_tasks=n_tasks,
                )
                last_snap = _snapshot_from_state(state)
                stacked_metrics = jax.tree.map(
                    lambda a, b: jnp.concatenate([a, b[None]]),
                    normal_metrics, restructure_metrics,
                )
                stacked_snaps = tuple(
                    jnp.concatenate([ns, ls[None]], axis=0)
                    for ns, ls in zip(normal_snaps, last_snap)
                )
                return state, (stacked_metrics, stacked_snaps)

            def scan_steps(state, data):
                images, labels = data
                images = images.reshape(
                    n_inner_blocks, prune_frequency, *images.shape[1:])
                labels = labels.reshape(
                    n_inner_blocks, prune_frequency, *labels.shape[1:])
                state, (metrics, snaps) = jax.lax.scan(
                    _inner_step, state, (images, labels))
                metrics = jax.tree.map(
                    lambda x: x.reshape(-1, *x.shape[2:]), metrics)
                snaps = tuple(
                    s.reshape(-1, *s.shape[2:]) for s in snaps)
                return state, (metrics, snaps)
        else:
            # Standard scan with restructure, no snapshots
            def _inner_step(state, data_block):
                normal_data = (data_block[0][:-1], data_block[1][:-1])
                state, normal_metrics = jax.lax.scan(
                    _normal_step, state, normal_data, unroll=SCAN_UNROLL)
                state, restructure_metrics = train_step(
                    state, (data_block[0][-1], data_block[1][-1]),
                    do_restructure=True, num_classes=num_classes, n_tasks=n_tasks,
                )
                stacked = jax.tree.map(
                    lambda a, b: jnp.concatenate([a, b[None]]),
                    normal_metrics, restructure_metrics,
                )
                return state, stacked

            def scan_steps(state, data):
                images, labels = data
                images = images.reshape(
                    n_inner_blocks, prune_frequency, *images.shape[1:])
                labels = labels.reshape(
                    n_inner_blocks, prune_frequency, *labels.shape[1:])
                state, metrics = jax.lax.scan(
                    _inner_step, state, (images, labels))
                metrics = jax.tree.map(
                    lambda x: x.reshape(-1, *x.shape[2:]), metrics)
                return state, metrics
    else:
        # No restructure — simple scan (utility_comparison variant)
        def scan_steps(state, data):
            images, labels = data
            state, metrics = jax.lax.scan(
                _normal_step, state, (images, labels), unroll=SCAN_UNROLL)
            return state, metrics

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses, all_accuracies = [], []
    all_per_seed_losses, all_per_seed_accuracies = [], []
    all_test_losses, all_test_accuracies = [], []
    all_snapshots = []  # list of (snap_tuple, step_indices)

    logging_active = cfg.get('mlflow', False) or cfg.get('wandb', False)
    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []

    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    for scan_idx in range(num_scans):
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))
        labels = jnp.array(np.stack([b[1] for b in batch]))

        scan_result = vmapped_scan(train_state, (images, labels))

        if collect_snapshots:
            train_state, (metrics, snapshots_jax) = scan_result
            # Transfer snapshots to numpy and subsample
            snap_np = tuple(np.array(v) for v in snapshots_jax)
            step_base = scan_idx * log_freq
            snap_sub, step_indices = _subsample_snapshots(
                snap_np, step_base, log_freq, snapshot_subsample)
            all_snapshots.append((snap_sub, step_indices))
        else:
            train_state, metrics = scan_result

        per_seed_loss = metrics.loss.mean(axis=1)
        per_seed_acc = metrics.correct.mean(axis=1)
        mean_loss = float(per_seed_loss.mean())
        mean_acc = float(per_seed_acc.mean())

        step = int(train_state.step[0].item())
        structure_metrics = compute_structure_metrics(train_state)
        structure_metrics.update(compute_column_metrics(train_state, n_tasks))

        if metrics.pruned_per_layer.size > 0:
            pruned = np.array(metrics.pruned_per_layer.sum(axis=1))
            generated = np.array(metrics.generated_per_layer.sum(axis=1))
            n_layers = pruned.shape[1]
            structure_metrics['total_pruned'] = float(pruned.sum(axis=1).mean())
            structure_metrics['total_generated'] = float(generated.sum(axis=1).mean())
            for l in range(n_layers):
                structure_metrics[f'layer_{l}/pruned'] = float(pruned[:, l].mean())
                structure_metrics[f'layer_{l}/generated'] = float(generated[:, l].mean())

        # Tagged vs untagged scalar metrics (mixed_generation)
        if collect_snapshots:
            structure_metrics.update(
                _compute_tag_metrics(train_state, n_tasks))

        test_metrics_dict = {}
        if eval_freq > 0 and step % eval_freq == 0 and test_data is not None:
            t_imgs, t_lbls = streams[0].get_test_batch()
            test_loss, test_acc = evaluate_test(
                train_state.model, t_imgs, t_lbls, num_classes, n_tasks,
            )
            mean_test_loss = float(test_loss.mean())
            mean_test_acc = float(test_acc.mean())
            all_test_losses.append(mean_test_loss)
            all_test_accuracies.append(mean_test_acc)
            test_metrics_dict = {
                'test_loss': mean_test_loss,
                'test_accuracy': mean_test_acc,
            }

        if logging_active:
            def _log(ml, sl, ma, sa, psl, psa, sm, tm, s):
                base = {'loss': ml, 'loss_std': sl, 'accuracy': ma, 'accuracy_std': sa}
                base.update(sm)
                base.update(tm)
                log_metrics(base, cfg, step=s)
                log_child_metrics({'loss': psl, 'accuracy': psa}, cfg, step=s)

            log_futures.append(log_executor.submit(
                _log,
                mean_loss, float(per_seed_loss.std()),
                mean_acc, float(per_seed_acc.std()),
                per_seed_loss.tolist(), per_seed_acc.tolist(),
                structure_metrics, test_metrics_dict, step,
            ))

        all_losses.append(mean_loss)
        all_accuracies.append(mean_acc)
        all_per_seed_losses.append(np.array(per_seed_loss))
        all_per_seed_accuracies.append(np.array(per_seed_acc))

        pbar.update(log_freq)
        postfix = {'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'}
        if test_metrics_dict:
            postfix['t_acc'] = f'{test_metrics_dict["test_accuracy"]:.4f}'
        pbar.set_postfix(postfix)

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()

    return (train_state, all_losses, all_accuracies,
            all_per_seed_losses, all_per_seed_accuracies,
            all_test_losses, all_test_accuracies,
            all_snapshots)


# ---------------------------------------------------------------------------
# End-of-training logging
# ---------------------------------------------------------------------------

def log_utility_distributions(train_state, streams, cfg, n_tasks):
    """Compute and log utility distribution plots as plotly artifacts."""
    import plotly.graph_objects as go
    import mlflow

    # Use EMA utility from tracker (consistent with GIF and scalar logging)
    utility = np.array(train_state.structure_tracker.unit_stats.utility)  # (n_seeds, L, U)
    unit_mask = np.array(train_state.model.unit_mask)                     # (n_seeds, L, U)

    has_tags = hasattr(train_state.structure_tracker, 'column_tag')
    if has_tags:
        column_tag = np.array(train_state.structure_tracker.column_tag)

    n_seeds = utility.shape[0]
    n_layers = utility.shape[1]

    for s in range(n_seeds):
        for l in range(n_layers):
            active = unit_mask[s, l].astype(bool)
            if not active.any():
                continue

            fig = go.Figure()
            # Compute shared bin edges from all active units
            all_vals = utility[s, l][active]
            n_bins = 50
            bin_lo, bin_hi = float(all_vals.min()), float(all_vals.max())
            bin_size = max((bin_hi - bin_lo) / n_bins, 1e-8)

            if has_tags:
                tag_groups = [
                    (0, 'Initial', 'gray'),
                    (1, 'Column-constrained', 'blue'),
                    (2, 'Free', 'red'),
                ]
                for tag_val, tag_name, tag_color in tag_groups:
                    mask = active & (column_tag[s, l] == tag_val)
                    if mask.any():
                        fig.add_trace(go.Histogram(
                            x=utility[s, l][mask], name=tag_name,
                            marker_color=tag_color, opacity=0.6,
                            xbins=dict(start=bin_lo, end=bin_hi, size=bin_size)))
                fig.update_layout(barmode='overlay')
            else:
                fig.add_trace(go.Histogram(
                    x=all_vals, name='All units',
                    marker_color='steelblue',
                    xbins=dict(start=bin_lo, end=bin_hi, size=bin_size)))

            fig.update_layout(
                title=f'EMA Utility Distribution — Seed {cfg.seed[s]}, Layer {l}',
                xaxis_title='EMA Utility',
                yaxis_title='Count',
            )
            mlflow.log_figure(
                fig, artifact_file=f'seed_{s}/utility_dist_layer_{l}.html')

    # Log scalar summary metrics
    for l in range(n_layers):
        active_utils = []
        for s in range(n_seeds):
            active = unit_mask[s, l].astype(bool)
            if active.any():
                active_utils.append(utility[s, l][active])
        if active_utils:
            all_utils = np.concatenate(active_utils)
            log_metrics({
                f'final/layer_{l}/mean_utility': float(all_utils.mean()),
                f'final/layer_{l}/median_utility': float(np.median(all_utils)),
                f'final/layer_{l}/sum_utility': float(all_utils.sum() / n_seeds),
            }, cfg)


def _save_snapshots(all_snapshots, cfg):
    """Concatenate and save snapshot data as mlflow artifact."""
    import mlflow
    import tempfile

    names = ['utility', 'unit_mask', 'age', 'column_tag']
    full_data = {}
    for i, name in enumerate(names):
        full_data[name] = np.concatenate(
            [s[0][i] for s in all_snapshots], axis=1)
    full_data['step_indices'] = np.concatenate(
        [s[1] for s in all_snapshots])

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez_compressed(f.name, **full_data)
        mlflow.log_artifact(f.name, artifact_path='snapshots')
    os.unlink(f.name)
    print(f'Saved unit snapshots: {full_data["utility"].shape[1]} timesteps, '
          f'{full_data["utility"].shape[0]} seeds')

    # Generate utility distribution GIF from snapshot data
    _save_utility_distribution_gif(full_data, cfg)


def _save_utility_distribution_gif(full_data, cfg, n_frames=60, seed_idx=0):
    """Create a GIF showing utility distribution evolution over training."""
    import mlflow
    import tempfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    utility = full_data['utility'][seed_idx]       # (T, L, U)
    unit_mask = full_data['unit_mask'][seed_idx]    # (T, L, U)
    column_tag = full_data['column_tag'][seed_idx]  # (T, L, U)
    step_indices = full_data['step_indices']        # (T,)

    T, n_layers, max_units = utility.shape
    has_tags = column_tag.max() > 0

    # Sample n_frames evenly spaced timesteps
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    # Compute shared x-axis range from all timesteps
    all_active_utils = utility[unit_mask.astype(bool)]
    if all_active_utils.size == 0:
        return
    x_lo = float(np.percentile(all_active_utils, 0.5))
    x_hi = float(np.percentile(all_active_utils, 99.5))
    n_bins = 40
    bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)

    tag_config = [
        (0, 'Initial', 'gray', 0.5),
        (1, 'Column-constr.', '#1e64dc', 0.7),
        (2, 'Free', '#dc321e', 0.7),
    ] if has_tags else [
        (-1, 'All units', 'steelblue', 0.8),
    ]

    for l in range(n_layers):
        # Pre-compute y-axis max across all frames for this layer
        all_maxes = []
        for t in frame_indices:
            active = unit_mask[t, l].astype(bool)
            vals = utility[t, l][active]
            if vals.size > 0:
                counts, _ = np.histogram(vals, bins=bin_edges)
                all_maxes.append(counts.max())
        y_max = int(np.percentile(all_maxes, 95) * 1.15) if all_maxes else 1

        fig, ax = plt.subplots(figsize=(8, 4))

        def animate(frame_num, layer=l, ym=y_max):
            ax.clear()
            t = frame_indices[frame_num]
            step = int(step_indices[t])
            active = unit_mask[t, layer].astype(bool)

            if has_tags:
                for tag_val, tag_name, color, alpha in tag_config:
                    mask = active & (column_tag[t, layer] == tag_val)
                    vals = utility[t, layer][mask]
                    if vals.size > 0:
                        ax.hist(vals, bins=bin_edges, color=color, alpha=alpha,
                                label=f'{tag_name} ({vals.size})')
            else:
                vals = utility[t, layer][active]
                if vals.size > 0:
                    ax.hist(vals, bins=bin_edges, color='steelblue', alpha=0.8,
                            label=f'All ({vals.size})')

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(0, ym)
            ax.set_xlabel('EMA Utility')
            ax.set_ylabel('Count')
            ax.set_title(f'Layer {layer} — Step {step}')
            ax.legend(loc='upper right', fontsize=8)

        anim = FuncAnimation(fig, animate, frames=n_frames, interval=150)
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            anim.save(f.name, writer='pillow', dpi=100)
            mlflow.log_artifact(f.name, artifact_path='gifs')
            print(f'Saved utility distribution GIF: layer {l}')
        os.unlink(f.name)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='../conf/column_guided', config_name='stationary', version_base='1.1')
def main(cfg: DictConfig) -> None:
    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)

    if cfg.seed is None:
        cfg.seed = [np.random.randint(0, 1_000_000_000)]
    elif isinstance(cfg.seed, int):
        cfg.seed = [cfg.seed]
    else:
        cfg.seed = list(cfg.seed)

    if cfg.get('log_individual_seeds', False) and not cfg.get('mlflow', False):
        raise ValueError(
            'log_individual_seeds requires mlflow. '
            'Set mlflow=true or disable log_individual_seeds.'
        )

    n_tasks = cfg.dataset.get('n_tasks', 5)
    set_seed(cfg.seed[0])
    init_child_runs(cfg.seed, cfg)

    train_state, streams, n_params, num_classes = prepare_experiment(cfg, n_tasks)

    test_data = None
    if cfg.train.get('eval_freq', 0) > 0:
        test_data = True  # signals run_experiment to use stream's get_test_batch

    (train_state, all_losses, all_accuracies,
     all_per_seed_losses, all_per_seed_accuracies,
     all_test_losses, all_test_accuracies,
     all_snapshots) = run_experiment(
        cfg, train_state, streams, num_classes, n_tasks, test_data=test_data,
    )

    average_loss = float(np.mean(all_losses))
    average_accuracy = float(np.mean(all_accuracies))
    n_tail = max(1, len(all_losses) // 10)
    asymptotic_loss = float(np.mean(all_losses[-n_tail:]))
    asymptotic_accuracy = float(np.mean(all_accuracies[-n_tail:]))

    print(f'Average loss:       {average_loss:.4f}')
    print(f'Average accuracy:   {average_accuracy:.4f}')
    print(f'Asymptotic loss:    {asymptotic_loss:.4f}')
    print(f'Asymptotic accuracy:{asymptotic_accuracy:.4f}')

    summary = {
        'average_loss': average_loss,
        'average_accuracy': average_accuracy,
        'asymptotic_loss': asymptotic_loss,
        'asymptotic_accuracy': asymptotic_accuracy,
        'num_params': n_params,
    }
    if all_test_losses:
        n_test_tail = max(1, len(all_test_losses) // 10)
        summary['asymptotic_test_loss'] = float(np.mean(all_test_losses[-n_test_tail:]))
        summary['asymptotic_test_accuracy'] = float(np.mean(all_test_accuracies[-n_test_tail:]))
        print(f'Asymptotic test accuracy: {summary["asymptotic_test_accuracy"]:.4f}')

    log_metrics(summary, cfg)

    if all_per_seed_losses:
        per_seed_losses = np.stack(all_per_seed_losses)
        per_seed_accs = np.stack(all_per_seed_accuracies)
        log_child_metrics({
            'average_loss': per_seed_losses.mean(axis=0).tolist(),
            'average_accuracy': per_seed_accs.mean(axis=0).tolist(),
            'asymptotic_loss': per_seed_losses[-n_tail:].mean(axis=0).tolist(),
            'asymptotic_accuracy': per_seed_accs[-n_tail:].mean(axis=0).tolist(),
            'num_params': [n_params] * len(cfg.seed),
        }, cfg)

    # End-of-training utility distribution logging
    variant = cfg.get('variant', 'column_guided')
    if variant in ('utility_comparison', 'mixed_generation'):
        log_utility_distributions(train_state, streams, cfg, n_tasks)

    # Save snapshot data for mixed_generation
    if all_snapshots and cfg.get('mlflow', False):
        _save_snapshots(all_snapshots, cfg)

    finish_child_runs(cfg)
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
