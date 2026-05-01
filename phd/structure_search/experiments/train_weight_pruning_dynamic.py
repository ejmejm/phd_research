"""Sparse weight-pruning training using DynamicNetwork.

Functionally identical to ``train_weight_pruning.py`` (same loss, same
EMA contribution utility, same prune-then-grow cycle, same diagnostics)
but stores W1 in per-unit sparse form (``input_indices`` of length
``max_connections_per_unit`` per unit) rather than the dense
``(max_hidden, input_dim)`` of ``PaddedMLP``. Saves memory/compute when
the per-unit fan-in is much smaller than the input dimension (e.g.,
random_sparse and structure_search with many tasks).
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from typing import Optional, Tuple

# Resolve relative MLflow tracking URI before Hydra changes CWD
_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
if _uri.startswith('sqlite:///') and not os.path.isabs(_uri[len('sqlite:///'):]):
    os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{os.path.abspath(_uri[len("sqlite:///"):])}'

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import (
    prepare_optimizer, set_seed, rng_from_string,
)
from phd.jax_core.models import lecun_uniform
from phd.jax_core.optimizers import EqxOptimizer
from phd.jax_core.optimizers.adam import AdamState
from phd.jax_core.utils import configure_jax, count_params, stack_pytrees, tree_replace
from phd.research_utils.logging import (
    init_experiment, init_child_runs, import_logger, bind_to_active_run,
    log_metrics, log_child_metrics, finish_child_runs, finish_experiment,
)
from phd.structure_search.data import load_dataset, ParallelMNISTStream
from phd.structure_search.dynamic_network import (
    DynamicNetwork, build_outgoing_indices, sync_outgoing_weights,
)


SCAN_UNROLL = 4
HIDDEN_LAYER = 0  # this script uses a single hidden layer (max_layers=1)


def _lecun_uniform_dyn(key, shape, fan_in):
    """Tracer-friendly LeCun uniform: ``fan_in`` may be a traced array.

    The cached ``phd.jax_core.models.lecun_uniform`` declares ``in_dim``
    static, which fails under vmap/scan when ``fan_in`` is computed from
    a sampled scalar. This local copy keeps ``fan_in`` purely dynamic.
    """
    bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in, 1.0).astype(jnp.float32))
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * bound


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StructureModel(eqx.Module):
    """DynamicNetwork + EMA per-connection utility tracking.

    ``network`` holds the trainable params (``weights``, ``output_weights``)
    and the structure arrays (``input_indices``, ``unit_mask``,
    ``output_mask``, ``outgoing_*``). The two utility arrays mirror the
    weight shapes and are updated by the EMA in ``train_step``.
    """
    network: DynamicNetwork
    w_in_utility: jax.Array   # (max_layers, max_units_per_layer, max_connections_per_unit)
    w_out_utility: jax.Array  # (output_dim, buffer_size)

    def __call__(self, x):
        return self.network(x)


def _model_filter_spec(model: StructureModel):
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(
        lambda m: (m.network.weights, m.network.output_weights),
        spec, (True, True),
    )


# ---------------------------------------------------------------------------
# Initial-structure builders (one per init_strategy)
# ---------------------------------------------------------------------------

def _init_dense_structure(network, key, initial, input_dim, output_dim, max_conns):
    kw1, kw2 = jax.random.split(key)

    used_mask = (jnp.arange(max_conns) < input_dim)
    row_indices = jnp.where(
        used_mask, jnp.arange(max_conns, dtype=jnp.int32), jnp.int32(-1),
    )
    new_input_indices = network.input_indices.at[HIDDEN_LAYER, :initial].set(
        jnp.broadcast_to(row_indices, (initial, max_conns)),
    )

    W1 = lecun_uniform(kw1, (initial, max_conns), in_dim=input_dim)
    W1 = W1 * used_mask.astype(jnp.float32)
    new_weights = network.weights.at[HIDDEN_LAYER, :initial].set(W1)

    new_unit_mask = network.unit_mask.at[HIDDEN_LAYER, :initial].set(1)

    new_output_mask = network.output_mask.at[:, input_dim:input_dim + initial].set(1)
    W2 = lecun_uniform(kw2, (output_dim, initial), in_dim=initial)
    new_output_weights = network.output_weights.at[:, input_dim:input_dim + initial].set(W2)

    return tree_replace(
        network,
        input_indices=new_input_indices,
        weights=new_weights,
        unit_mask=new_unit_mask,
        output_mask=new_output_mask,
        output_weights=new_output_weights,
    )


def _init_block_sparse_structure(
    network, key, initial, input_dim, output_dim, max_conns, n_tasks,
):
    kw1, kw2 = jax.random.split(key)

    input_per_task = input_dim // n_tasks
    output_per_task = output_dim // n_tasks
    units_per_task = initial // n_tasks

    used_mask = (jnp.arange(max_conns) < input_per_task)
    base_row = jnp.where(
        used_mask, jnp.arange(max_conns, dtype=jnp.int32), jnp.int32(-1),
    )

    W1_full = lecun_uniform(kw1, (initial, max_conns), in_dim=input_per_task)
    W1_full = W1_full * used_mask.astype(jnp.float32)
    W2_full = lecun_uniform(kw2, (output_dim, initial), in_dim=units_per_task)

    new_input_indices = network.input_indices
    new_weights = network.weights
    new_output_mask = network.output_mask
    new_output_weights = network.output_weights

    for t in range(n_tasks):
        u0, u1 = t * units_per_task, (t + 1) * units_per_task
        i0 = t * input_per_task
        # Inputs for task-t units: i0..i0+input_per_task-1, padded with -1.
        row = jnp.where(used_mask, base_row + i0, jnp.int32(-1))
        new_input_indices = new_input_indices.at[HIDDEN_LAYER, u0:u1].set(
            jnp.broadcast_to(row, (units_per_task, max_conns)),
        )
        new_weights = new_weights.at[HIDDEN_LAYER, u0:u1].set(W1_full[u0:u1])

        # Output: outputs in task t connect to hidden units in task t.
        o0, o1 = t * output_per_task, (t + 1) * output_per_task
        new_output_mask = new_output_mask.at[
            o0:o1, input_dim + u0:input_dim + u1,
        ].set(1)
        new_output_weights = new_output_weights.at[
            o0:o1, input_dim + u0:input_dim + u1,
        ].set(W2_full[o0:o1, u0:u1])

    new_unit_mask = network.unit_mask.at[HIDDEN_LAYER, :initial].set(1)
    return tree_replace(
        network,
        input_indices=new_input_indices,
        weights=new_weights,
        unit_mask=new_unit_mask,
        output_mask=new_output_mask,
        output_weights=new_output_weights,
    )


def _init_random_sparse_structure(
    network, key, initial, input_dim, output_dim, max_conns, range_in, range_out,
):
    kw1, kw2, kperm_in, kperm_out, kk_in, kk_out = jax.random.split(key, 6)

    k_ins = jax.random.randint(kk_in, (initial,), range_in[0], range_in[1] + 1)
    k_outs = jax.random.randint(kk_out, (initial,), range_out[0], range_out[1] + 1)
    in_perm_keys = jax.random.split(kperm_in, initial)
    out_perm_keys = jax.random.split(kperm_out, initial)
    in_weight_keys = jax.random.split(kw1, initial)
    out_weight_keys = jax.random.split(kw2, initial)

    def per_unit(in_perm_key, out_perm_key, in_w_key, out_w_key, k_in, k_out):
        in_perm = jax.random.permutation(in_perm_key, input_dim)
        slot_used = (jnp.arange(max_conns) < k_in)
        idx_row = jnp.where(
            slot_used,
            jnp.take(in_perm, jnp.arange(max_conns), mode='clip'),
            jnp.int32(-1),
        ).astype(jnp.int32)
        w_row = _lecun_uniform_dyn(in_w_key, (max_conns,), k_in)
        w_row = w_row * slot_used.astype(jnp.float32)

        out_perm = jax.random.permutation(out_perm_key, output_dim)
        out_pos_used = (jnp.arange(output_dim) < k_out).astype(jnp.int32)
        out_mask_col = jnp.zeros(output_dim, dtype=jnp.int32).at[out_perm].set(out_pos_used)
        out_w_full = _lecun_uniform_dyn(out_w_key, (output_dim,), jnp.float32(1.0))
        out_w_col = out_w_full * out_mask_col.astype(jnp.float32)

        return idx_row, w_row, out_mask_col, out_w_col

    idx_rows, w_rows, out_mask_cols, out_w_cols = jax.vmap(per_unit)(
        in_perm_keys, out_perm_keys, in_weight_keys, out_weight_keys, k_ins, k_outs,
    )

    new_input_indices = network.input_indices.at[HIDDEN_LAYER, :initial].set(idx_rows)
    new_weights = network.weights.at[HIDDEN_LAYER, :initial].set(w_rows)
    new_unit_mask = network.unit_mask.at[HIDDEN_LAYER, :initial].set(1)
    new_output_mask = network.output_mask.at[:, input_dim:input_dim + initial].set(out_mask_cols.T)
    new_output_weights = network.output_weights.at[:, input_dim:input_dim + initial].set(out_w_cols.T)

    return tree_replace(
        network,
        input_indices=new_input_indices,
        weights=new_weights,
        unit_mask=new_unit_mask,
        output_mask=new_output_mask,
        output_weights=new_output_weights,
    )


def init_model(
    cfg: DictConfig, input_dim: int, output_dim: int, n_tasks: int,
    *, key: PRNGKeyArray,
) -> StructureModel:
    """Build a StructureModel per the configured init strategy."""
    max_hidden = int(cfg.model.max_hidden_units)
    initial = int(cfg.model.initial_hidden_units)
    strategy = cfg.model.init_strategy
    activation = cfg.model.activation

    # Default ``max_connections_per_unit`` to ``input_dim`` so existing
    # configs still work; sparse runs should override this in the YAML.
    max_conns = int(cfg.model.get('max_connections_per_unit', input_dim))
    max_fan_out = int(cfg.model.get('max_fan_out', max_hidden))

    network = DynamicNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=1,
        max_units_per_layer=max_hidden,
        max_connections_per_unit=max_conns,
        activations=(activation,),
        max_fan_out=max_fan_out,
        init_strategy='empty',
        key=key,
    )

    if initial > 0:
        if strategy == 'dense':
            assert max_conns >= input_dim, (
                f'max_connections_per_unit ({max_conns}) must be >= input_dim '
                f'({input_dim}) for dense init'
            )
            network = _init_dense_structure(
                network, key, initial, input_dim, output_dim, max_conns,
            )
        elif strategy == 'block_sparse':
            assert n_tasks > 1, 'block_sparse init requires n_tasks > 1'
            assert initial % n_tasks == 0, 'initial_hidden_units must be divisible by n_tasks'
            input_per_task = input_dim // n_tasks
            assert max_conns >= input_per_task, (
                f'max_connections_per_unit ({max_conns}) must be >= '
                f'input_dim/n_tasks ({input_per_task}) for block_sparse init'
            )
            network = _init_block_sparse_structure(
                network, key, initial, input_dim, output_dim, max_conns, n_tasks,
            )
        elif strategy == 'random_sparse':
            range_in = tuple(cfg.model.init_random_range_in)
            range_out = tuple(cfg.model.init_random_range_out)
            assert max_conns >= range_in[1], (
                f'max_connections_per_unit ({max_conns}) must be >= '
                f'init_random_range_in upper ({range_in[1]})'
            )
            network = _init_random_sparse_structure(
                network, key, initial, input_dim, output_dim,
                max_conns, range_in, range_out,
            )
        else:
            raise ValueError(f'Unknown init_strategy: {strategy}')

    network = build_outgoing_indices(network)

    return StructureModel(
        network=network,
        w_in_utility=jnp.zeros_like(network.weights),
        w_out_utility=jnp.zeros_like(network.output_weights),
    )


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: StructureModel
    optimizer: EqxOptimizer
    step: jax.Array
    rng: PRNGKeyArray


# ---------------------------------------------------------------------------
# Optimizer state reset (Adam: each weight has its own age)
# ---------------------------------------------------------------------------

def _reset_optimizer_at(
    optimizer: EqxOptimizer,
    weights_reset: jax.Array,
    output_weights_reset: jax.Array,
) -> EqxOptimizer:
    """Zero per-parameter Adam state at positions where the reset masks are True."""
    state = optimizer.state
    if not isinstance(state, AdamState):
        return optimizer

    def _reset_field(field, fill_dtype):
        new_w = jnp.where(
            weights_reset, jnp.asarray(0, dtype=fill_dtype), field.network.weights,
        )
        new_ow = jnp.where(
            output_weights_reset, jnp.asarray(0, dtype=fill_dtype), field.network.output_weights,
        )
        return eqx.tree_at(
            lambda f: (f.network.weights, f.network.output_weights),
            field, (new_w, new_ow),
        )

    new_state = AdamState(
        lr=state.lr,
        step=_reset_field(state.step, jnp.int32),
        exp_avg=_reset_field(state.exp_avg, jnp.float32),
        exp_avg_sq=_reset_field(state.exp_avg_sq, jnp.float32),
    )
    return tree_replace(optimizer, state=new_state)


# ---------------------------------------------------------------------------
# Restructure: prune connections, then generate units
# ---------------------------------------------------------------------------

def _median_active_util(model: StructureModel) -> jax.Array:
    """Median of utilities over currently-active connections (NaN-safe)."""
    util = jnp.concatenate([
        model.w_in_utility.reshape(-1), model.w_out_utility.reshape(-1),
    ])
    mask_in = (model.network.input_indices >= 0).astype(jnp.float32).reshape(-1)
    mask_out = model.network.output_mask.astype(jnp.float32).reshape(-1)
    mask = jnp.concatenate([mask_in, mask_out])
    masked = jnp.where(mask > 0, util, jnp.nan)
    med = jnp.nanmedian(masked)
    return jnp.where(jnp.isnan(med), 0.0, med)


def _active_connection_count(network: DynamicNetwork) -> jax.Array:
    n_in = (network.input_indices >= 0).sum()
    n_out = network.output_mask.sum()
    return (n_in + n_out).astype(jnp.float32)


def _prune_connections(model: StructureModel, prune_count: int):
    """Prune the ``prune_count`` smallest-utility active connections.

    Returns ``(new_model, pruned_w1, pruned_w2, n_pruned)``. The
    ``pruned_*`` boolean arrays mark positions that were just zeroed and
    are used to reset optimizer state.
    """
    network = model.network
    n_in = model.w_in_utility.size

    util_flat = jnp.concatenate([
        model.w_in_utility.reshape(-1),
        model.w_out_utility.reshape(-1),
    ])
    mask_in_flat = (network.input_indices >= 0).astype(jnp.float32).reshape(-1)
    mask_out_flat = network.output_mask.astype(jnp.float32).reshape(-1)
    mask_flat = jnp.concatenate([mask_in_flat, mask_out_flat])

    score = jnp.where(mask_flat > 0, util_flat, jnp.inf)
    _, smallest_idx = jax.lax.top_k(-score, prune_count)
    is_active_pick = jnp.isfinite(score[smallest_idx])
    n_pruned = is_active_pick.sum().astype(jnp.int32)

    prune_flat = jnp.zeros_like(mask_flat).at[smallest_idx].set(
        is_active_pick.astype(jnp.float32),
    )
    pruned_in = prune_flat[:n_in].reshape(model.w_in_utility.shape)
    pruned_out = prune_flat[n_in:].reshape(model.w_out_utility.shape)

    keep_in = 1.0 - pruned_in
    keep_out = 1.0 - pruned_out

    new_weights = network.weights * keep_in
    new_input_indices = jnp.where(pruned_in > 0, jnp.int32(-1), network.input_indices)
    new_output_weights = network.output_weights * keep_out
    new_output_mask = (
        network.output_mask.astype(jnp.float32) * keep_out
    ).astype(network.output_mask.dtype)
    new_w_in_util = model.w_in_utility * keep_in
    new_w_out_util = model.w_out_utility * keep_out

    # Deactivate units with no remaining incoming or outgoing connections
    has_in = (new_input_indices >= 0).any(axis=-1)               # (max_layers, max_units)
    input_dim = network.input_dim
    max_units = network.max_units_per_layer
    hidden_block = new_output_mask[:, input_dim:input_dim + max_units]   # (output_dim, max_units)
    has_out_layer0 = (hidden_block > 0).any(axis=0)                       # (max_units,)
    has_out = jnp.zeros_like(network.unit_mask, dtype=jnp.bool_)
    has_out = has_out.at[HIDDEN_LAYER].set(has_out_layer0)
    new_unit_mask = (
        network.unit_mask.astype(jnp.bool_) & has_in & has_out
    ).astype(network.unit_mask.dtype)

    new_network = tree_replace(
        network,
        weights=new_weights,
        input_indices=new_input_indices,
        output_weights=new_output_weights,
        output_mask=new_output_mask,
        unit_mask=new_unit_mask,
    )
    new_model = tree_replace(
        model,
        network=new_network,
        w_in_utility=new_w_in_util,
        w_out_utility=new_w_out_util,
    )
    return new_model, pruned_in.astype(bool), pruned_out.astype(bool), n_pruned


def _generate_units(
    model: StructureModel, optimizer: EqxOptimizer, rng: PRNGKeyArray,
    *, max_units: int, range_in: Tuple[int, int],
    range_out: Tuple[int, int], connection_budget: float,
) -> Tuple[StructureModel, EqxOptimizer, jax.Array]:
    """Generate up to ``max_units`` new hidden units, capped by the budget.

    A new unit is generated only if the network currently has room for
    one with the worst-case fan-in/fan-out (``range_in[1] + range_out[1]``)
    without exceeding ``connection_budget`` total active connections.
    """
    max_cost = float(range_in[1] + range_out[1])
    median_util = _median_active_util(model)
    cur_conns_init = _active_connection_count(model.network)
    network0 = model.network
    max_conns = network0.max_connections_per_unit
    input_dim = network0.input_dim
    output_dim = network0.output_dim
    max_units_per_layer = network0.max_units_per_layer

    def step(carry, key):
        model, optimizer, cur_conns, n_gen = carry
        network = model.network
        keys = jax.random.split(key, 5)
        k_in_key, k_out_key, in_key, out_key, w_in_key = keys

        layer_unit_mask = network.unit_mask[HIDDEN_LAYER]
        free_slot = jnp.argmax(layer_unit_mask == 0)
        has_slot = jnp.any(layer_unit_mask == 0)
        can_afford = (cur_conns + max_cost) <= connection_budget
        do_gen = has_slot & can_afford

        k_in = jax.random.randint(k_in_key, (), range_in[0], range_in[1] + 1)
        k_out = jax.random.randint(k_out_key, (), range_out[0], range_out[1] + 1)

        # Sample input indices: take first k_in entries of an input permutation.
        in_perm = jax.random.permutation(in_key, input_dim)
        slot_used = (jnp.arange(max_conns) < k_in)
        new_idx_row = jnp.where(
            slot_used,
            jnp.take(in_perm, jnp.arange(max_conns), mode='clip'),
            jnp.int32(-1),
        ).astype(jnp.int32)
        new_w_row = _lecun_uniform_dyn(w_in_key, (max_conns,), k_in)
        new_w_row = new_w_row * slot_used.astype(jnp.float32)

        # Sample output positions (output weights init to zero so no shock).
        out_perm = jax.random.permutation(out_key, output_dim)
        out_pos_used = (jnp.arange(output_dim) < k_out).astype(network.output_mask.dtype)
        new_out_mask_col = jnp.zeros(output_dim, dtype=network.output_mask.dtype).at[
            out_perm,
        ].set(out_pos_used)

        buffer_pos = input_dim + HIDDEN_LAYER * max_units_per_layer + free_slot

        prev_idx_row = network.input_indices[HIDDEN_LAYER, free_slot]
        prev_w_row = network.weights[HIDDEN_LAYER, free_slot]
        prev_unit = network.unit_mask[HIDDEN_LAYER, free_slot]
        prev_out_mask_col = network.output_mask[:, buffer_pos]
        prev_ow_col = network.output_weights[:, buffer_pos]
        prev_w_in_util_row = model.w_in_utility[HIDDEN_LAYER, free_slot]
        prev_w_out_util_col = model.w_out_utility[:, buffer_pos]

        new_input_indices = network.input_indices.at[HIDDEN_LAYER, free_slot].set(
            jnp.where(do_gen, new_idx_row, prev_idx_row),
        )
        new_weights = network.weights.at[HIDDEN_LAYER, free_slot].set(
            jnp.where(do_gen, new_w_row, prev_w_row),
        )
        new_unit_mask = network.unit_mask.at[HIDDEN_LAYER, free_slot].set(
            jnp.where(do_gen, jnp.asarray(1, dtype=network.unit_mask.dtype), prev_unit),
        )
        new_output_mask = network.output_mask.at[:, buffer_pos].set(
            jnp.where(do_gen, new_out_mask_col, prev_out_mask_col),
        )
        new_output_weights = network.output_weights.at[:, buffer_pos].set(
            jnp.where(do_gen, jnp.zeros(output_dim, dtype=prev_ow_col.dtype), prev_ow_col),
        )
        new_w_in_util = model.w_in_utility.at[HIDDEN_LAYER, free_slot].set(
            jnp.where(do_gen,
                      slot_used.astype(jnp.float32) * median_util,
                      prev_w_in_util_row),
        )
        new_w_out_util = model.w_out_utility.at[:, buffer_pos].set(
            jnp.where(do_gen,
                      new_out_mask_col.astype(jnp.float32) * median_util,
                      prev_w_out_util_col),
        )

        new_network = tree_replace(
            network,
            input_indices=new_input_indices,
            weights=new_weights,
            unit_mask=new_unit_mask,
            output_mask=new_output_mask,
            output_weights=new_output_weights,
        )
        new_model = tree_replace(
            model,
            network=new_network,
            w_in_utility=new_w_in_util,
            w_out_utility=new_w_out_util,
        )

        # Reset optimizer state at newly active positions
        weights_reset = jnp.zeros_like(network.weights, dtype=bool)
        weights_reset = weights_reset.at[HIDDEN_LAYER, free_slot].set(slot_used & do_gen)
        ow_reset = jnp.zeros_like(network.output_weights, dtype=bool)
        ow_reset = ow_reset.at[:, buffer_pos].set((new_out_mask_col > 0) & do_gen)
        new_optimizer = _reset_optimizer_at(optimizer, weights_reset, ow_reset)

        cost = (k_in + k_out).astype(jnp.float32)
        new_cur_conns = jnp.where(do_gen, cur_conns + cost, cur_conns)
        new_n_gen = n_gen + do_gen.astype(jnp.int32)

        return (new_model, new_optimizer, new_cur_conns, new_n_gen), None

    keys = jax.random.split(rng, max_units)
    (model, optimizer, _, n_generated), _ = jax.lax.scan(
        step, (model, optimizer, cur_conns_init, jnp.int32(0)), keys,
    )
    return model, optimizer, n_generated


def restructure(
    state: TrainState, prune_count: int, max_units_per_event: int,
    range_in: Tuple[int, int], range_out: Tuple[int, int],
    connection_budget: float,
) -> Tuple[TrainState, jax.Array, jax.Array]:
    new_model, pruned_in, pruned_out, n_pruned = _prune_connections(
        state.model, prune_count,
    )
    optimizer = _reset_optimizer_at(state.optimizer, pruned_in, pruned_out)

    next_rng, gen_rng = jax.random.split(state.rng)
    new_model, optimizer, n_generated = _generate_units(
        new_model, optimizer, gen_rng,
        max_units=max_units_per_event, range_in=range_in, range_out=range_out,
        connection_budget=connection_budget,
    )

    # Outgoing indices were invalidated by pruning (input_indices set to -1)
    # and generation (new input_indices written). Rebuild + sync once here.
    new_network = build_outgoing_indices(new_model.network)
    new_model = tree_replace(new_model, network=new_network)

    return tree_replace(
        state, model=new_model, optimizer=optimizer, rng=next_rng,
    ), n_pruned, n_generated


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def train_step(
    state: TrainState, data, *, num_classes: int, n_tasks: int, decay: float,
):
    images, labels = data                          # labels: (B, K), images: (B, K*C_in)
    one_hot = jax.nn.one_hot(labels, num_classes)   # (B, K, C)

    def loss_fn(model):
        outputs, buffers = jax.vmap(model)(images)
        outputs_r = outputs.reshape(-1, n_tasks, num_classes)
        log_probs = jax.nn.log_softmax(outputs_r, axis=-1)
        loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        return loss, (outputs_r, buffers)

    (loss, (outputs_r, buffers)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(state.model)
    correct = (jnp.argmax(outputs_r, axis=-1) == labels).astype(jnp.float32).mean()

    updates, new_optimizer = state.optimizer.with_update(grads, state.model)
    new_model = eqx.apply_updates(state.model, updates)

    # Outgoing weights are captured (under stop_gradient) inside __call__.
    # They must mirror the freshly-updated incoming weights for the next step.
    new_network = sync_outgoing_weights(new_model.network)
    new_model = tree_replace(new_model, network=new_network)

    # EMA of contribution utility = |source activation| * |weight|, gated by mask.
    network = new_model.network
    buffer_abs = jnp.abs(buffers).mean(axis=0)                # (buffer_size,)

    input_idx = network.input_indices
    conn_mask_in = (input_idx >= 0).astype(jnp.float32)
    safe_idx = jnp.maximum(input_idx, 0)
    src_abs_in = buffer_abs[safe_idx]                          # (L, U, C)
    w_in_util = (decay * new_model.w_in_utility
                 + (1 - decay) * jnp.abs(network.weights) * src_abs_in) * conn_mask_in

    out_mask_f = network.output_mask.astype(jnp.float32)
    w_out_util = (decay * new_model.w_out_utility
                  + (1 - decay) * jnp.abs(network.output_weights) * buffer_abs[None, :]) * out_mask_f

    new_state = tree_replace(
        state,
        model=tree_replace(new_model, w_in_utility=w_in_util, w_out_utility=w_out_util),
        optimizer=new_optimizer,
        step=state.step + 1,
    )
    return new_state, jnp.stack([loss, correct])


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def _standardize(images: np.ndarray, ref: Optional[np.ndarray] = None):
    src = ref if ref is not None else images
    mean = src.mean(axis=0, keepdims=True)
    std = src.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (images - mean) / std


def prepare_experiment(cfg: DictConfig):
    seeds = cfg.seed
    n_tasks = int(cfg.task.n_tasks)
    permute_period = int(cfg.task.permute_period)
    standardize = bool(cfg.task.get('standardize', False))

    images, labels, num_classes, input_dim_per_task = load_dataset('mnist', split='train')
    if standardize:
        images = _standardize(images)
    input_dim = n_tasks * input_dim_per_task
    output_dim = n_tasks * num_classes

    streams, train_states = [], []
    for seed in seeds:
        rng = jax.random.key(seed)
        streams.append(ParallelMNISTStream(
            images=images, labels=labels, n_tasks=n_tasks,
            batch_size=cfg.train.batch_size, seed=seed,
            permute_period=permute_period,
        ))

        model = init_model(
            cfg, input_dim, output_dim, n_tasks,
            key=rng_from_string(rng, 'model'),
        )
        optimizer = prepare_optimizer(
            model, cfg.optimizer.name, cfg.optimizer,
            filter_spec=_model_filter_spec(model),
        )
        train_states.append(TrainState(
            model=model, optimizer=optimizer,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
        ))

    n_params = count_params(train_states[0].model)
    network0 = train_states[0].model.network
    n_active_conns = int(
        (network0.input_indices >= 0).sum() + network0.output_mask.sum()
    )
    initial_units = int(network0.unit_mask.sum())
    print(f'StructureModel: padded_params={n_params}, '
          f'active_connections={n_active_conns}, initial_units={initial_units}, '
          f'seeds={seeds}')

    return stack_pytrees(train_states), streams, num_classes, n_tasks


# ---------------------------------------------------------------------------
# Structure diagnostics (computed outside JIT, on the vmapped model)
# ---------------------------------------------------------------------------

def _structure_diagnostics(model: StructureModel, n_tasks: int) -> dict:
    """Per-seed structure statistics from the stacked model.

    Path-purity (when ``n_tasks > 1``) is the per-output mean of
    ``#same-task-paths / #total-paths``. A path is an
    ``(output, hidden, input)`` triple with active output and input
    connections; multiple hidden units between the same ``(output,
    input)`` pair count as multiple paths. Outputs with no paths are
    excluded from the mean.
    """
    network = model.network
    unit_active = network.unit_mask[..., HIDDEN_LAYER, :].astype(jnp.float32)  # (S, U)

    input_idx = network.input_indices[..., HIDDEN_LAYER, :, :]    # (S, U, C)
    fan_in_per_unit = (input_idx >= 0).sum(axis=-1).astype(jnp.float32)   # (S, U)

    input_dim = network.input_dim
    max_units = network.max_units_per_layer
    output_dim = network.output_dim
    hidden_out_mask = network.output_mask[..., :, input_dim:input_dim + max_units].astype(jnp.float32)
    fan_out_per_unit = hidden_out_mask.sum(axis=-2)                # (S, U)

    n_active_units = jnp.maximum(unit_active.sum(axis=-1), 1.0)
    n_in_active = (network.input_indices >= 0).sum(axis=(-1, -2, -3)).astype(jnp.float32)
    n_out_active = network.output_mask.sum(axis=(-1, -2)).astype(jnp.float32)

    stats = {
        'active_units': unit_active.sum(axis=-1),
        'active_connections': n_in_active + n_out_active,
        'mean_fan_in': (fan_in_per_unit * unit_active).sum(axis=-1) / n_active_units,
        'mean_fan_out': (fan_out_per_unit * unit_active).sum(axis=-1) / n_active_units,
    }

    if n_tasks > 1:
        input_per_task = input_dim // n_tasks
        output_per_task = output_dim // n_tasks

        # Per-(seed, unit) fan-in count broken down by source task.
        # Use bincount with -1 connections sent to a dummy bucket and dropped.
        def _unit_fan_in_per_task(idx_row):
            bucket = jnp.where(idx_row >= 0, idx_row // input_per_task, n_tasks)
            return jnp.bincount(bucket, length=n_tasks + 1)[:n_tasks].astype(jnp.float32)

        fan_in_per_task = jax.vmap(jax.vmap(_unit_fan_in_per_task))(input_idx)  # (S, U, T)
        total_fan_in_per_unit = fan_in_per_task.sum(axis=-1)                     # (S, U)

        # total_per_output[s, o] = sum_u out_h[s, o, u] * total_fan_in_per_unit[s, u]
        total_per_output = jnp.einsum(
            '...ou,...u->...o', hidden_out_mask, total_fan_in_per_unit,
        )

        # same_per_output[s, o] = sum_u out_h[s, o, u] * fan_in_per_task[s, u, task(o)]
        output_task = jnp.arange(output_dim) // output_per_task              # (output_dim,)
        fan_in_at_o = jnp.take(fan_in_per_task, output_task, axis=-1)        # (S, U, output_dim)
        fan_in_at_o = jnp.swapaxes(fan_in_at_o, -1, -2)                       # (S, output_dim, U)
        same_per_output = (hidden_out_mask * fan_in_at_o).sum(axis=-1)        # (S, output_dim)

        output_active = total_per_output > 0
        purity_per_output = jnp.where(
            output_active,
            same_per_output / jnp.maximum(total_per_output, 1.0),
            0.0,
        )
        sum_purity = purity_per_output.sum(axis=-1)
        n_active_outputs = output_active.sum(axis=-1).astype(jnp.float32)
        stats['path_purity'] = sum_purity / jnp.maximum(n_active_outputs, 1.0)
    else:
        stats['path_purity'] = jnp.ones_like(stats['active_units'])

    return stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig, train_state: TrainState, streams,
                   num_classes: int, n_tasks: int):
    log_freq = int(cfg.train.log_freq)
    num_log_periods = int(cfg.train.total_steps) // log_freq
    structure_search_enabled = bool(cfg.structure_search.enabled)
    prune_freq = int(cfg.structure_search.prune_frequency) if structure_search_enabled else log_freq

    train_step_fn = partial(
        train_step, num_classes=num_classes, n_tasks=n_tasks,
        decay=float(cfg.structure_search.decay_rate),
    )

    if structure_search_enabled:
        assert log_freq % prune_freq == 0, \
            f'log_freq={log_freq} must be divisible by prune_frequency={prune_freq}'
        prune_cycles_per_log = log_freq // prune_freq
        restructure_fn = partial(
            restructure,
            prune_count=int(cfg.structure_search.prune_count),
            max_units_per_event=int(cfg.structure_search.max_units_per_event),
            range_in=tuple(cfg.structure_search.gen_range_in),
            range_out=tuple(cfg.structure_search.gen_range_out),
            connection_budget=float(cfg.structure_search.connection_budget),
        )

        def prune_cycle(state, cycle_data):
            state, metrics = jax.lax.scan(train_step_fn, state, cycle_data, unroll=SCAN_UNROLL)
            state, n_pruned, n_generated = restructure_fn(state)
            return state, (metrics, n_pruned, n_generated)

        def scan_log_period(state, data):
            data = jax.tree.map(
                lambda x: x.reshape(prune_cycles_per_log, prune_freq, *x.shape[1:]), data,
            )
            state, (metrics, pruned, generated) = jax.lax.scan(prune_cycle, state, data)
            return state, metrics.reshape(-1, *metrics.shape[2:]), pruned, generated
    else:
        def scan_log_period(state, data):
            state, metrics = jax.lax.scan(train_step_fn, state, data, unroll=SCAN_UNROLL)
            return state, metrics, None, None

    vmapped_scan = jax.jit(jax.vmap(scan_log_period))

    all_losses, all_accs, all_per_seed_losses, all_per_seed_accs = [], [], [], []
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))
    cumulative_pruned = 0.0
    cumulative_generated = 0.0

    for _ in range(num_log_periods):
        batch = [s.sample_batch(log_freq) for s in streams]
        imgs = jnp.array(np.stack([b[0] for b in batch]))
        lbls = jnp.array(np.stack([b[1] for b in batch]))

        train_state, metrics, pruned, generated = vmapped_scan(train_state, (imgs, lbls))

        per_seed_loss = metrics[..., 0].mean(axis=1)
        per_seed_acc = metrics[..., 1].mean(axis=1)
        mean_loss, mean_acc = float(per_seed_loss.mean()), float(per_seed_acc.mean())
        std_loss, std_acc = float(per_seed_loss.std()), float(per_seed_acc.std())
        step = int(train_state.step[0].item())

        structure_metrics = {
            k: float(v.mean()) for k, v in
            _structure_diagnostics(train_state.model, n_tasks).items()
        }
        if pruned is not None:
            cumulative_pruned += float(pruned.sum(axis=1).mean())
            cumulative_generated += float(generated.sum(axis=1).mean())
            structure_metrics['cumulative_pruned'] = cumulative_pruned
            structure_metrics['cumulative_generated'] = cumulative_generated

        if logging_active:
            log_futures.append(log_executor.submit(
                _bg_log, mean_loss, std_loss, mean_acc, std_acc,
                per_seed_loss.tolist(), per_seed_acc.tolist(),
                structure_metrics, cfg, step,
            ))

        all_losses.append(mean_loss)
        all_accs.append(mean_acc)
        all_per_seed_losses.append(np.array(per_seed_loss))
        all_per_seed_accs.append(np.array(per_seed_acc))

        pbar.update(log_freq)
        pbar.set_postfix(loss=f'{mean_loss:.4f}', acc=f'{mean_acc:.4f}',
                         units=f'{structure_metrics["active_units"]:.0f}',
                         conn=f'{structure_metrics["active_connections"]:.0f}')

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()
    return train_state, all_losses, all_accs, all_per_seed_losses, all_per_seed_accs


def _bg_log(mean_loss, std_loss, mean_acc, std_acc,
            per_seed_loss, per_seed_acc, structure_metrics, cfg, step):
    metrics = {
        'loss': mean_loss, 'loss_std': std_loss,
        'accuracy': mean_acc, 'accuracy_std': std_acc,
    }
    metrics.update(structure_metrics)
    log_metrics(metrics, cfg, step=step)
    log_child_metrics(
        {'loss': per_seed_loss, 'accuracy': per_seed_acc}, cfg, step=step,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_config(cfg: DictConfig) -> dict:
    configure_jax(cfg)
    import_logger(cfg)
    bind_to_active_run(cfg)

    if cfg.seed is None:
        cfg.seed = [int(np.random.randint(0, 1_000_000_000))]
    elif isinstance(cfg.seed, int):
        cfg.seed = [cfg.seed]
    else:
        cfg.seed = list(cfg.seed)

    if cfg.get('log_individual_seeds', False) and not cfg.get('mlflow', False):
        raise ValueError('log_individual_seeds requires mlflow logging.')

    set_seed(cfg.seed[0])
    init_child_runs(cfg.seed, cfg)

    train_state, streams, num_classes, n_tasks = prepare_experiment(cfg)
    train_state, all_losses, all_accs, all_per_seed_losses, all_per_seed_accs = run_experiment(
        cfg, train_state, streams, num_classes, n_tasks,
    )

    n_tail = max(1, len(all_losses) // 10)
    final_network = train_state.model.network
    summary = {
        'average_loss': float(np.mean(all_losses)),
        'asymptotic_loss': float(np.mean(all_losses[-n_tail:])),
        'asymptotic_accuracy': float(np.mean(all_accs[-n_tail:])),
        'final_active_units': float(
            final_network.unit_mask.sum(axis=(-1, -2)).mean()
        ),
        'final_active_connections': float(
            ((final_network.input_indices >= 0).sum(axis=(-1, -2, -3))
             + final_network.output_mask.sum(axis=(-1, -2))).mean()
        ),
    }
    print(f'Average loss: {summary["average_loss"]:.4f} | '
          f'Asymptotic loss: {summary["asymptotic_loss"]:.4f} | '
          f'Asymptotic acc: {summary["asymptotic_accuracy"]:.4f}')
    log_metrics(summary, cfg)

    if all_per_seed_losses:
        per_seed_losses = np.stack(all_per_seed_losses)
        per_seed_accs = np.stack(all_per_seed_accs)
        log_child_metrics({
            'average_loss': per_seed_losses.mean(axis=0).tolist(),
            'asymptotic_loss': per_seed_losses[-n_tail:].mean(axis=0).tolist(),
            'asymptotic_accuracy': per_seed_accs[-n_tail:].mean(axis=0).tolist(),
        }, cfg)

    finish_child_runs(cfg)
    return summary


@hydra.main(config_path='../conf', config_name='train_weight_pruning', version_base='1.1')
def main(cfg: DictConfig) -> None:
    cfg = init_experiment(cfg.project, cfg)
    try:
        run_config(cfg)
    finally:
        finish_experiment(cfg)


if __name__ == '__main__':
    main()
