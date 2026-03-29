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
    ConnectivityManager, UnitStats, contribution_utility, median_utility_init,
)
from phd.structure_search.data import load_dataset, ParallelMNISTStream
from phd.structure_search.dynamic_network import (
    DynamicNetwork, sync_outgoing_weights, build_outgoing_indices,
    init_random_dynamic_network, count_active_connections, count_active_units,
)
from phd.structure_search.metrics import StepMetrics, compute_structure_metrics


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Column-aware utility
# ---------------------------------------------------------------------------

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

        tracker = ConnectivityManager(
            model=model,
            prune_rate=cfg.structure_tracker.prune_rate,
            connection_budget=cfg.structure_tracker.connection_budget,
            decay_rate=cfg.structure_tracker.decay_rate,
            maturity_threshold=cfg.structure_tracker.maturity_threshold,
            max_new_units_per_step=cfg.structure_tracker.get('max_new_units_per_step', 512),
            output_connect_strategy='all',  # column_generate handles outputs internally; 'random_sparse' not used
            output_weight_init='zero',
            utility_fn=partial(column_utility, n_tasks=n_tasks),
            generate_fn=partial(column_generate, n_tasks=n_tasks),
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

def run_experiment(cfg, train_state, streams, num_classes, n_tasks, test_data=None):
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq
    prune_frequency = cfg.structure_tracker.get('prune_frequency', log_freq)
    eval_freq = cfg.train.get('eval_freq', 0)

    assert log_freq % prune_frequency == 0, (
        f'log_freq ({log_freq}) must be divisible by prune_frequency ({prune_frequency})'
    )
    n_inner_blocks = log_freq // prune_frequency

    def _normal_step(state, data):
        return train_step(state, data, do_restructure=False,
                          num_classes=num_classes, n_tasks=n_tasks)

    def _inner_step(state, data_block):
        normal_data = (data_block[0][:-1], data_block[1][:-1])
        state, normal_metrics = jax.lax.scan(_normal_step, state, normal_data, unroll=SCAN_UNROLL)
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
        images = images.reshape(n_inner_blocks, prune_frequency, *images.shape[1:])
        labels = labels.reshape(n_inner_blocks, prune_frequency, *labels.shape[1:])
        state, metrics = jax.lax.scan(_inner_step, state, (images, labels))
        metrics = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), metrics)
        return state, metrics

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses, all_accuracies = [], []
    all_per_seed_losses, all_per_seed_accuracies = [], []
    all_test_losses, all_test_accuracies = [], []

    logging_active = cfg.get('mlflow', False) or cfg.get('wandb', False)
    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []

    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    for scan_idx in range(num_scans):
        batch = [stream.sample_batch(log_freq) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))
        labels = jnp.array(np.stack([b[1] for b in batch]))

        train_state, metrics = vmapped_scan(train_state, (images, labels))

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
            all_test_losses, all_test_accuracies)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='../conf', config_name='config', version_base='1.1')
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
     all_test_losses, all_test_accuracies) = run_experiment(
        cfg, train_state, streams, num_classes, n_tasks, test_data=test_data,
    )

    average_loss = float(np.mean(all_losses))
    n_tail = max(1, len(all_losses) // 10)
    asymptotic_loss = float(np.mean(all_losses[-n_tail:]))
    asymptotic_accuracy = float(np.mean(all_accuracies[-n_tail:]))

    print(f'Average loss:       {average_loss:.4f}')
    print(f'Asymptotic loss:    {asymptotic_loss:.4f}')
    print(f'Asymptotic accuracy:{asymptotic_accuracy:.4f}')

    summary = {
        'average_loss': average_loss,
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
            'asymptotic_loss': per_seed_losses[-n_tail:].mean(axis=0).tolist(),
            'asymptotic_accuracy': per_seed_accs[-n_tail:].mean(axis=0).tolist(),
            'num_params': [n_params] * len(cfg.seed),
        }, cfg)

    finish_child_runs(cfg)
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
