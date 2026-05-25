"""Sparse Evolutionary Training (SET) on DynamicNetwork.

Implements the algorithm of Mocanu et al. (Nature Comms 2018, arXiv:1707.04780):

    1. Initialize each bipartite layer (W1: input->hidden, W2: hidden->output)
       as an Erdos-Renyi random graph with edge probability
       p(W_ij) = epsilon * (n_in + n_out) / (n_in * n_out).
    2. Periodically (every ``evolve_frequency`` steps), per layer:
         - Remove a fraction zeta of the smallest-positive active weights.
         - Remove a fraction zeta of the largest-negative active weights
           (i.e. the negatives closest to zero).
         - Regrow the same number of new connections at random inactive
           positions, restricted to the same row -- per-unit fan-in is
           preserved (matches the per-unit sparse storage of DynamicNetwork
           and keeps the layer's connection count exactly constant).
    3. Hidden units stay permanently active; only connections evolve.

Defaults from the paper: epsilon=20, zeta=0.3.

The hidden layer width is *derived* from the connection budget:

    expected_connections(hidden) = epsilon * (input_dim + output_dim)
                                 + 2 * epsilon * hidden
    hidden = floor((budget / epsilon - input_dim - output_dim) / 2)

Reuses the training stack of ``train_weight_pruning_dynamic.py``: parallel
multi-task MNIST stream, vmapped multi-seed scan, Adam with per-weight age
reset on prune/regrow events, MLflow logging.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from typing import Optional

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
HIDDEN_LAYER = 0  # single hidden layer (max_layers=1)
DIVERGENCE_LOSS_THRESHOLD = 1e6  # stop training early if train loss exceeds this (or NaN/inf)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StructureModel(eqx.Module):
    """Thin wrapper over DynamicNetwork (no utility tracking under SET)."""
    network: DynamicNetwork

    def __call__(self, x):
        return self.network(x)


def _model_filter_spec(model: StructureModel):
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(
        lambda m: (m.network.weights, m.network.output_weights),
        spec, (True, True),
    )


# ---------------------------------------------------------------------------
# Erdos-Renyi initialization
# ---------------------------------------------------------------------------

def _init_erdos_renyi_structure(
    network, key, hidden, input_dim, output_dim, max_conns, p_w1, p_w2,
):
    """Build the initial sparse topology for both bipartite layers.

    `p_w1` and `p_w2` are the (pre-clipped) per-edge inclusion probabilities
    for each bipartite layer. Clipped to [0, 1] before sampling.
    """
    k_w1_mask, k_w1_w, k_w2_mask, k_w2_w = jax.random.split(key, 4)

    # --- W1 (input -> hidden) ---
    p_w1 = jnp.minimum(jnp.asarray(p_w1, dtype=jnp.float32), 1.0)
    mask_w1 = jax.random.bernoulli(k_w1_mask, p_w1, (hidden, input_dim))

    # Per row, pack active column indices into the leading slots; pad with -1.
    cols = jnp.broadcast_to(
        jnp.arange(input_dim, dtype=jnp.int32), mask_w1.shape,
    )
    score = jnp.where(mask_w1, cols, input_dim + cols)  # actives sort first
    sort_order = jnp.argsort(score, axis=-1)
    sorted_cols = jnp.take_along_axis(cols, sort_order, axis=-1)
    sorted_active = jnp.take_along_axis(mask_w1, sort_order, axis=-1)
    take_cols = sorted_cols[:, :max_conns]
    take_active = sorted_active[:, :max_conns]
    new_idx_row = jnp.where(take_active, take_cols, jnp.int32(-1))

    fan_in_per_row = take_active.sum(axis=-1).astype(jnp.float32)
    raw_w = jax.random.uniform(k_w1_w, (hidden, max_conns), minval=-1.0, maxval=1.0)
    bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in_per_row, 1.0))
    w1 = raw_w * bound[:, None] * take_active.astype(jnp.float32)

    new_input_indices = network.input_indices.at[HIDDEN_LAYER, :hidden].set(new_idx_row)
    new_weights = network.weights.at[HIDDEN_LAYER, :hidden].set(w1)
    new_unit_mask = network.unit_mask.at[HIDDEN_LAYER, :hidden].set(1)

    # --- W2 (hidden -> output) ---
    p_w2 = jnp.minimum(jnp.asarray(p_w2, dtype=jnp.float32), 1.0)
    mask_w2 = jax.random.bernoulli(k_w2_mask, p_w2, (output_dim, hidden))
    fan_in_w2 = mask_w2.sum(axis=-1).astype(jnp.float32)
    raw_w2 = jax.random.uniform(k_w2_w, (output_dim, hidden), minval=-1.0, maxval=1.0)
    bound_w2 = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in_w2, 1.0))
    w2 = raw_w2 * bound_w2[:, None] * mask_w2.astype(jnp.float32)

    new_output_mask = network.output_mask.at[:, input_dim:input_dim + hidden].set(
        mask_w2.astype(network.output_mask.dtype),
    )
    new_output_weights = network.output_weights.at[:, input_dim:input_dim + hidden].set(w2)

    return tree_replace(
        network,
        input_indices=new_input_indices,
        weights=new_weights,
        unit_mask=new_unit_mask,
        output_mask=new_output_mask,
        output_weights=new_output_weights,
    )


def init_model(
    cfg: DictConfig, input_dim: int, output_dim: int,
    hidden_units: int, max_conns: int, max_fan_out: int,
    p_w1: float, p_w2: float,
    *, key: PRNGKeyArray,
) -> StructureModel:
    activation = cfg.model.activation

    network = DynamicNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=1,
        max_units_per_layer=hidden_units,
        max_connections_per_unit=max_conns,
        activations=(activation,),
        max_fan_out=max_fan_out,
        init_strategy='empty',
        key=key,
    )

    network = _init_erdos_renyi_structure(
        network, key, hidden_units, input_dim, output_dim, max_conns, p_w1, p_w2,
    )
    network = build_outgoing_indices(network)
    return StructureModel(network=network)


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    model: StructureModel
    optimizer: EqxOptimizer
    step: jax.Array
    rng: PRNGKeyArray
    # Contribution-utility traces (EMA of |source_activation| * |weight|) and
    # per-weight step counters for Adam-style bias correction. Always present
    # and updated; only consulted by the prune step when `set.prune_metric ==
    # 'utility'`. For 'magnitude' (default) the prune step ignores them and
    # behavior is bit-equivalent to the original SET implementation.
    util_w1: jax.Array       # same shape as model.network.weights
    util_w2: jax.Array       # same shape as model.network.output_weights
    util_step_w1: jax.Array  # int32, same shape as util_w1
    util_step_w2: jax.Array  # int32, same shape as util_w2


# ---------------------------------------------------------------------------
# Adam state reset (each weight has its own age)
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
# SET prune-and-regrow
# ---------------------------------------------------------------------------

def _signed_prune_threshold(weights, active, zeta):
    """Compute (prune_mask) selecting smallest-positive and largest-negative
    fractions zeta of active weights, separately by sign.

    "Largest negative" means the negatives closest to zero. The two thresholds
    are derived from a global per-layer sort, so pruning is across the full
    weight matrix (not per row).
    """
    flat = weights.reshape(-1)
    flat_active = active.reshape(-1)

    n_pos = jnp.sum(flat_active & (flat > 0))
    n_neg = jnp.sum(flat_active & (flat < 0))
    n_prune_pos = (zeta * n_pos.astype(jnp.float32)).astype(jnp.int32)
    n_prune_neg = (zeta * n_neg.astype(jnp.float32)).astype(jnp.int32)

    pos_score = jnp.where(flat_active & (flat > 0), flat, jnp.inf)
    sorted_pos = jnp.sort(pos_score)
    threshold_pos = sorted_pos[jnp.maximum(n_prune_pos - 1, 0)]
    prune_pos_flat = (flat_active & (flat > 0)
                      & (flat <= threshold_pos)
                      & (n_prune_pos > 0))

    # For negatives: rank by -w ascending == by closeness to zero descending.
    # We want negatives whose value is closest to zero, i.e. largest among
    # the negatives. Equivalently: smallest -w.
    neg_score = jnp.where(flat_active & (flat < 0), -flat, jnp.inf)
    sorted_neg = jnp.sort(neg_score)
    threshold_negabs = sorted_neg[jnp.maximum(n_prune_neg - 1, 0)]
    prune_neg_flat = (flat_active & (flat < 0)
                      & (-flat <= threshold_negabs)
                      & (n_prune_neg > 0))

    return (prune_pos_flat | prune_neg_flat).reshape(weights.shape)


def _utility_prune_threshold(score, active, zeta):
    """Prune the smallest-utility zeta fraction of active weights, sign-agnostic.

    Used when `set.prune_metric == 'utility'`. The score is the bias-corrected
    EMA `|source_activation| * |weight|` (already non-negative).
    """
    flat = score.reshape(-1)
    flat_active = active.reshape(-1)

    n_active = jnp.sum(flat_active)
    n_prune = (zeta * n_active.astype(jnp.float32)).astype(jnp.int32)

    masked_score = jnp.where(flat_active, flat, jnp.inf)
    sorted_score = jnp.sort(masked_score)
    threshold = sorted_score[jnp.maximum(n_prune - 1, 0)]
    prune_flat = flat_active & (flat <= threshold) & (n_prune > 0)
    return prune_flat.reshape(score.shape)


def _bias_corrected_utility(util, step, decay):
    """Adam-style bias correction: u / (1 - decay**step), with a floor on the
    denominator so freshly-regrown weights (step=0) don't divide by zero."""
    step_f = jnp.maximum(step.astype(jnp.float32), 1.0)
    correction = jnp.maximum(1.0 - jnp.power(decay, step_f), 1e-12)
    return util / correction


def _set_evolve_w1(network, optimizer, util_w1, util_step_w1, rng, zeta, *,
                   prune_metric, utility_decay):
    """Prune + per-row random regrow on W1."""
    weights = network.weights[HIDDEN_LAYER]            # (U, C)
    idx = network.input_indices[HIDDEN_LAYER]          # (U, C)
    active = idx >= 0
    U, C = weights.shape
    input_dim = network.input_dim

    fan_in_pre = active.sum(axis=-1).astype(jnp.float32)  # (U,)

    if prune_metric == 'utility':
        score = _bias_corrected_utility(
            util_w1[HIDDEN_LAYER], util_step_w1[HIDDEN_LAYER], utility_decay,
        )
        prune_mask = _utility_prune_threshold(score, active, zeta)
    else:
        prune_mask = _signed_prune_threshold(weights, active, zeta)

    weights_after = jnp.where(prune_mask, 0.0, weights)
    idx_after = jnp.where(prune_mask, jnp.int32(-1), idx)
    n_pruned_per_row = prune_mask.sum(axis=-1)            # (U,)

    row_keys = jax.random.split(rng, U)

    def regrow_row(row_idx, row_w, n_to_regrow, fan_in_target, key):
        active_slot = row_idx >= 0
        # Build "in_use" over input_dim. Inactive slots write True at column 0
        # (a harmless overwrite -- if there is any active slot at column 0,
        # active_slot=True there overrides). Use the per-slot "active" value
        # so inactive slots write False.
        safe_col = jnp.where(active_slot, row_idx, 0)
        in_use = jnp.zeros(input_dim, dtype=jnp.bool_).at[safe_col].set(active_slot)

        col_key, w_key = jax.random.split(key)
        col_score = jax.random.uniform(col_key, (input_dim,))
        col_score = jnp.where(in_use, -jnp.inf, col_score)
        sort_order = jnp.argsort(-col_score)              # top-k inactive first

        free_slot = ~active_slot                          # (C,)
        free_rank = jnp.cumsum(free_slot.astype(jnp.int32)) - 1
        slot_takes = free_slot & (free_rank >= 0) & (free_rank < n_to_regrow)
        safe_rank = jnp.clip(free_rank, 0, input_dim - 1)
        chosen_col = sort_order[safe_rank]                # (C,)
        new_row_idx = jnp.where(slot_takes, chosen_col.astype(jnp.int32), row_idx)

        # New weights via LeCun uniform with the post-regrow per-row fan-in
        # (== pre-prune per-row fan-in, since regrowth replaces pruned slots).
        new_w_raw = jax.random.uniform(w_key, (C,), minval=-1.0, maxval=1.0)
        bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in_target, 1.0))
        new_w_vals = new_w_raw * bound
        new_row_w = jnp.where(slot_takes, new_w_vals, row_w)
        return new_row_idx, new_row_w, slot_takes

    new_idx, new_w, regrow_mask = jax.vmap(regrow_row)(
        idx_after, weights_after, n_pruned_per_row, fan_in_pre, row_keys,
    )

    new_input_indices = network.input_indices.at[HIDDEN_LAYER].set(new_idx)
    new_weights_full = network.weights.at[HIDDEN_LAYER].set(new_w)

    weights_reset = jnp.zeros_like(network.weights, dtype=bool)
    weights_reset = weights_reset.at[HIDDEN_LAYER].set(prune_mask | regrow_mask)

    new_network = tree_replace(
        network, input_indices=new_input_indices, weights=new_weights_full,
    )
    new_optimizer = _reset_optimizer_at(
        optimizer, weights_reset,
        jnp.zeros_like(network.output_weights, dtype=bool),
    )

    # Reset utility trace + per-weight step counter at every modified slot.
    modified = prune_mask | regrow_mask  # (U, C)
    new_util_w1 = util_w1.at[HIDDEN_LAYER].set(
        jnp.where(modified, jnp.float32(0.0), util_w1[HIDDEN_LAYER])
    )
    new_util_step_w1 = util_step_w1.at[HIDDEN_LAYER].set(
        jnp.where(modified, jnp.int32(0), util_step_w1[HIDDEN_LAYER])
    )

    return (new_network, new_optimizer, new_util_w1, new_util_step_w1,
            prune_mask.sum().astype(jnp.int32),
            regrow_mask.sum().astype(jnp.int32))


def _set_evolve_w2(network, optimizer, util_w2, util_step_w2, rng, zeta, *,
                   prune_metric, utility_decay):
    """Prune + per-row random regrow on W2."""
    input_dim = network.input_dim
    U = network.max_units_per_layer
    s, e = input_dim, input_dim + U

    weights_full = network.output_weights                 # (O, B)
    mask_full = network.output_mask.astype(jnp.bool_)
    weights = weights_full[:, s:e]                        # (O, U)
    active = mask_full[:, s:e]

    fan_in_pre = active.sum(axis=-1).astype(jnp.float32)  # (O,)

    if prune_metric == 'utility':
        util_slice = util_w2[:, s:e]
        step_slice = util_step_w2[:, s:e]
        score = _bias_corrected_utility(util_slice, step_slice, utility_decay)
        prune_mask = _utility_prune_threshold(score, active, zeta)
    else:
        prune_mask = _signed_prune_threshold(weights, active, zeta)

    weights_after = jnp.where(prune_mask, 0.0, weights)
    active_after = active & ~prune_mask
    n_pruned_per_row = prune_mask.sum(axis=-1)
    O = weights.shape[0]
    row_keys = jax.random.split(rng, O)

    def regrow_row(row_active, row_w, n_to_regrow, fan_in_target, key):
        col_key, w_key = jax.random.split(key)
        col_score = jax.random.uniform(col_key, (U,))
        col_score = jnp.where(row_active, -jnp.inf, col_score)
        sort_order = jnp.argsort(-col_score)
        rank = jnp.argsort(sort_order)                    # rank 0 = top
        slot_takes = (rank < n_to_regrow) & ~row_active
        new_w_raw = jax.random.uniform(w_key, (U,), minval=-1.0, maxval=1.0)
        bound = jnp.sqrt(3.0) / jnp.sqrt(jnp.maximum(fan_in_target, 1.0))
        new_w_vals = new_w_raw * bound
        new_row_w = jnp.where(slot_takes, new_w_vals, row_w)
        new_row_active = row_active | slot_takes
        return new_row_w, new_row_active, slot_takes

    new_w, new_active, regrow_mask = jax.vmap(regrow_row)(
        active_after, weights_after, n_pruned_per_row, fan_in_pre, row_keys,
    )

    new_output_weights = weights_full.at[:, s:e].set(new_w)
    new_output_mask = network.output_mask.at[:, s:e].set(
        new_active.astype(network.output_mask.dtype),
    )

    ow_reset = jnp.zeros_like(network.output_weights, dtype=bool)
    ow_reset = ow_reset.at[:, s:e].set(prune_mask | regrow_mask)

    new_network = tree_replace(
        network, output_weights=new_output_weights, output_mask=new_output_mask,
    )
    new_optimizer = _reset_optimizer_at(
        optimizer, jnp.zeros_like(network.weights, dtype=bool), ow_reset,
    )

    # Reset utility trace + per-weight step counter at every modified slot
    # within the hidden slice. Outside that slice the values stay at 0 anyway.
    modified_slice = prune_mask | regrow_mask  # (O, U)
    new_util_w2 = util_w2.at[:, s:e].set(
        jnp.where(modified_slice, jnp.float32(0.0), util_w2[:, s:e])
    )
    new_util_step_w2 = util_step_w2.at[:, s:e].set(
        jnp.where(modified_slice, jnp.int32(0), util_step_w2[:, s:e])
    )

    return (new_network, new_optimizer, new_util_w2, new_util_step_w2,
            prune_mask.sum().astype(jnp.int32),
            regrow_mask.sum().astype(jnp.int32))


def set_evolve(state: TrainState, *, zeta: float, prune_metric: str,
               utility_decay: float):
    """One SET prune-and-regrow event over both bipartite layers."""
    next_rng, k_w1, k_w2 = jax.random.split(state.rng, 3)

    network, optimizer, util_w1, util_step_w1, n_p1, n_r1 = _set_evolve_w1(
        state.model.network, state.optimizer, state.util_w1, state.util_step_w1,
        k_w1, zeta,
        prune_metric=prune_metric, utility_decay=utility_decay,
    )
    network, optimizer, util_w2, util_step_w2, n_p2, n_r2 = _set_evolve_w2(
        network, optimizer, state.util_w2, state.util_step_w2,
        k_w2, zeta,
        prune_metric=prune_metric, utility_decay=utility_decay,
    )
    network = build_outgoing_indices(network)

    new_state = tree_replace(
        state,
        model=tree_replace(state.model, network=network),
        optimizer=optimizer,
        rng=next_rng,
        util_w1=util_w1, util_w2=util_w2,
        util_step_w1=util_step_w1, util_step_w2=util_step_w2,
    )
    return new_state, n_p1, n_r1, n_p2, n_r2


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def train_step(state: TrainState, data, *, num_classes: int, n_tasks: int,
               utility_decay: float):
    images, labels = data
    one_hot = jax.nn.one_hot(labels, num_classes)

    def loss_fn(model):
        outputs, buffer = jax.vmap(model)(images)
        outputs_r = outputs.reshape(-1, n_tasks, num_classes)
        log_probs = jax.nn.log_softmax(outputs_r, axis=-1)
        loss_per_task = -jnp.sum(one_hot * log_probs, axis=-1)
        loss = jnp.mean(jnp.sum(loss_per_task, axis=1))
        return loss, (outputs_r, buffer)

    (loss, (outputs_r, buffer)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(state.model)
    correct = (jnp.argmax(outputs_r, axis=-1) == labels).astype(jnp.float32).mean()

    # ---- Utility trace EMA update (pre-step weights, current activations) ----
    # Done before the SGD apply so the trace uses |w| at the same state that
    # produced these activations. Inactive slots stay at 0 because either |w|=0
    # or the gather/mask zeros the source contribution.
    network = state.model.network
    input_dim = network.input_dim
    max_units = network.max_units_per_layer

    # W1 contribution: |x_i| * |w_{u,c}| where i = input_indices[u, c].
    input_indices = network.input_indices[HIDDEN_LAYER]                # (U, C)
    active_w1 = input_indices >= 0
    safe_idx = jnp.where(active_w1, input_indices, 0)
    x_per_slot = images[:, safe_idx]                                   # (B, U, C)
    x_per_slot_abs = jnp.where(active_w1, jnp.abs(x_per_slot), 0.0)
    x_mag = jnp.mean(x_per_slot_abs, axis=0)                           # (U, C)
    w1_abs = jnp.abs(network.weights[HIDDEN_LAYER])                    # (U, C)
    contrib_w1 = x_mag * w1_abs
    new_util_w1 = state.util_w1.at[HIDDEN_LAYER].set(
        utility_decay * state.util_w1[HIDDEN_LAYER]
        + (1.0 - utility_decay) * contrib_w1
    )
    new_util_step_w1 = state.util_step_w1.at[HIDDEN_LAYER].set(
        state.util_step_w1[HIDDEN_LAYER] + 1
    )

    # W2 contribution: |h_j| * |w_{k,j}| where h_j is the post-activation hidden
    # value at unit j. buffer is shape (B, buffer_size) with the hidden slice
    # at [:, input_dim:input_dim+max_units].
    s, e = input_dim, input_dim + max_units
    h_mag = jnp.mean(jnp.abs(buffer[:, s:e]), axis=0)                  # (U,)
    w2_hidden_abs = jnp.abs(state.model.network.output_weights[:, s:e])  # (O, U)
    contrib_w2 = h_mag[None, :] * w2_hidden_abs
    new_util_w2 = state.util_w2.at[:, s:e].set(
        utility_decay * state.util_w2[:, s:e]
        + (1.0 - utility_decay) * contrib_w2
    )
    new_util_step_w2 = state.util_step_w2.at[:, s:e].set(
        state.util_step_w2[:, s:e] + 1
    )

    updates, new_optimizer = state.optimizer.with_update(grads, state.model)
    new_model = eqx.apply_updates(state.model, updates)

    # Outgoing weights mirror the incoming weights for the next step's bwd.
    new_network = sync_outgoing_weights(new_model.network)
    new_model = tree_replace(new_model, network=new_network)

    new_state = tree_replace(
        state, model=new_model, optimizer=new_optimizer, step=state.step + 1,
        util_w1=new_util_w1, util_w2=new_util_w2,
        util_step_w1=new_util_step_w1, util_step_w2=new_util_step_w2,
    )
    return new_state, jnp.stack([loss, correct])


# ---------------------------------------------------------------------------
# Diagnostics, eval
# ---------------------------------------------------------------------------

def _standardize(images: np.ndarray, ref: Optional[np.ndarray] = None):
    src = ref if ref is not None else images
    mean = src.mean(axis=0, keepdims=True)
    std = src.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (images - mean) / std


def _structure_diagnostics(model: StructureModel, n_tasks: int) -> dict:
    """Per-seed structure statistics (active units, fan-in/out, path purity)."""
    network = model.network
    unit_active = network.unit_mask[..., HIDDEN_LAYER, :].astype(jnp.float32)
    input_idx = network.input_indices[..., HIDDEN_LAYER, :, :]
    fan_in_per_unit = (input_idx >= 0).sum(axis=-1).astype(jnp.float32)

    input_dim = network.input_dim
    max_units = network.max_units_per_layer
    output_dim = network.output_dim
    hidden_out_mask = network.output_mask[..., :, input_dim:input_dim + max_units].astype(jnp.float32)
    fan_out_per_unit = hidden_out_mask.sum(axis=-2)

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

        def _unit_fan_in_per_task(idx_row):
            bucket = jnp.where(idx_row >= 0, idx_row // input_per_task, n_tasks)
            return jnp.bincount(bucket, length=n_tasks + 1)[:n_tasks].astype(jnp.float32)

        fan_in_per_task = jax.vmap(jax.vmap(_unit_fan_in_per_task))(input_idx)
        total_fan_in_per_unit = fan_in_per_task.sum(axis=-1)

        total_per_output = jnp.einsum(
            '...ou,...u->...o', hidden_out_mask, total_fan_in_per_unit,
        )
        output_task = jnp.arange(output_dim) // output_per_task
        fan_in_at_o = jnp.take(fan_in_per_task, output_task, axis=-1)
        fan_in_at_o = jnp.swapaxes(fan_in_at_o, -1, -2)
        same_per_output = (hidden_out_mask * fan_in_at_o).sum(axis=-1)

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


def _eval_forward(model, images, labels, num_classes, n_tasks):
    outputs, _ = jax.vmap(model)(images)
    one_hot = jax.nn.one_hot(labels, num_classes)
    outputs_r = outputs.reshape(-1, n_tasks, num_classes)
    log_probs = jax.nn.log_softmax(outputs_r, axis=-1)
    loss = jnp.mean(jnp.sum(-jnp.sum(one_hot * log_probs, axis=-1), axis=1))
    correct = (jnp.argmax(outputs_r, axis=-1) == labels).astype(jnp.float32).mean()
    return loss, correct


def evaluate_test(batched_model, test_images, test_labels,
                  num_classes: int, n_tasks: int, batch_size: int = 512):
    @jax.jit
    def _eval_chunk(model, imgs, lbls):
        return jax.vmap(
            lambda m: _eval_forward(m, imgs, lbls, num_classes, n_tasks),
        )(model)

    n_test = test_images.shape[0]
    total_loss = total_acc = None
    n_chunks = 0
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        chunk_imgs = jnp.array(test_images[start:end])
        chunk_lbls = jnp.array(test_labels[start:end])
        cl, ca = _eval_chunk(batched_model, chunk_imgs, chunk_lbls)
        total_loss = cl if total_loss is None else total_loss + cl
        total_acc = ca if total_acc is None else total_acc + ca
        n_chunks += 1
    return total_loss / n_chunks, total_acc / n_chunks


# ---------------------------------------------------------------------------
# Sizing and experiment setup
# ---------------------------------------------------------------------------

def _derive_sizes(cfg: DictConfig, input_dim: int, output_dim: int):
    """Derive ``hidden_units``, ``max_connections_per_unit``, ``max_fan_out``,
    and the per-layer edge probabilities ``(p_w1, p_w2)`` from the SET budget
    configuration.

    Two init modes are supported (selected by ``cfg.set.init_mode``):

    - ``epsilon`` (default, original SET): user provides ``set.epsilon`` and
      ``set.connection_budget``; hidden width is derived as
      ``floor((budget/ε - input_dim - output_dim) / 2)``. Per-layer edge
      probabilities follow the Erdős-Rényi formula
      ``p_layer = ε(n+m)/(n·m)``, which gives *different* densities for W1
      and W2 when (n, m) are asymmetric (the typical case here).

    - ``uniform_p``: both bipartite layers share a single edge probability
      ``p = budget / ((input_dim + output_dim) · hidden_units)``. Hidden width
      is taken directly from ``target_hidden_units``. ``ε`` is not consulted.
      This keeps W1 and W2 at the *same* density at every scale, and yields
      constant per-hidden-unit W1 fan-in (``p·input_dim``) when budget and H
      are scaled proportionally.
    """
    init_mode = str(cfg.set.get('init_mode', 'epsilon'))
    budget = float(cfg.set.connection_budget)

    if init_mode == 'uniform_p':
        hidden = int(cfg.target_hidden_units)
        if hidden < 1:
            raise ValueError(
                f'target_hidden_units must be >= 1, got {hidden}'
            )
        p = budget / ((input_dim + output_dim) * hidden)
        if p > 1.0:
            raise ValueError(
                f'uniform_p init: budget={budget} requires p={p:.4f} > 1 at '
                f'hidden={hidden}, input={input_dim}, output={output_dim}. '
                f'Reduce budget or increase hidden_units.'
            )
        p_w1 = p
        p_w2 = p
        expected_w1_fan_in = p * input_dim
        expected_w1_col_fan = p * hidden
    elif init_mode == 'epsilon':
        epsilon = float(cfg.set.epsilon)
        hidden = int(np.floor(
            (budget / epsilon - input_dim - output_dim) / 2.0,
        ))
        if hidden < 1:
            raise ValueError(
                f'connection_budget={budget} with epsilon={epsilon}, '
                f'input_dim={input_dim}, output_dim={output_dim} yields '
                f'hidden_units={hidden}; budget too small.',
            )
        p_w1 = min(1.0, epsilon * (input_dim + hidden) / (input_dim * hidden))
        p_w2 = min(1.0, epsilon * (output_dim + hidden) / (output_dim * hidden))
        expected_w1_fan_in = epsilon * (input_dim + hidden) / hidden
        expected_w1_col_fan = epsilon * (input_dim + hidden) / input_dim
    else:
        raise ValueError(
            f"set.init_mode must be 'epsilon' or 'uniform_p', got {init_mode!r}"
        )

    safety = float(cfg.model.fan_in_safety_factor)
    max_conns = int(min(input_dim, max(1, np.ceil(safety * expected_w1_fan_in))))
    # max_fan_out caps the per-buffer-position outgoing array used by the
    # backward pass through the hidden layer. With one hidden layer it's
    # bounded by W1's expected column fan-in. We require at least 2x headroom
    # over the expected mean so SET has room for the column degree distribution
    # to grow (binomial → power-law).
    fan_out_factor = max(2.0, safety)
    max_fan_out = int(min(
        hidden,
        max(8, np.ceil(fan_out_factor * expected_w1_col_fan) + 4),
    ))
    return hidden, max_conns, max_fan_out, p_w1, p_w2


def prepare_experiment(cfg: DictConfig):
    seeds = cfg.seed
    n_tasks = int(cfg.task.n_tasks)
    permute_period = int(cfg.task.permute_period)
    standardize = bool(cfg.task.get('standardize', False))
    eval_freq = int(cfg.train.get('eval_freq', 0))

    raw_train_images, labels, num_classes, input_dim_per_task = load_dataset('mnist', split='train')
    images = _standardize(raw_train_images) if standardize else raw_train_images

    test_images = test_labels = None
    if eval_freq > 0:
        test_images, test_labels, _, _ = load_dataset('mnist', split='test')
        if standardize:
            test_images = _standardize(test_images, ref=raw_train_images)

    input_dim = n_tasks * input_dim_per_task
    output_dim = n_tasks * num_classes

    hidden_units, max_conns, max_fan_out, p_w1, p_w2 = _derive_sizes(
        cfg, input_dim, output_dim,
    )
    init_mode = str(cfg.set.get('init_mode', 'epsilon'))
    expected_w1 = p_w1 * input_dim * hidden_units
    expected_w2 = p_w2 * output_dim * hidden_units
    sizing_extra = (
        f'epsilon={float(cfg.set.epsilon):.4f}, '
        if init_mode == 'epsilon' else ''
    )
    print(
        f'SET sizing [init_mode={init_mode}]: {sizing_extra}'
        f'budget={cfg.set.connection_budget}, '
        f'input_dim={input_dim}, output_dim={output_dim} -> '
        f'hidden_units={hidden_units}, '
        f'p_w1={p_w1:.4f}, p_w2={p_w2:.4f}, '
        f'max_connections_per_unit={max_conns}, max_fan_out={max_fan_out} '
        f'(expected active conns: W1~{expected_w1:.0f}, W2~{expected_w2:.0f}, '
        f'total~{expected_w1 + expected_w2:.0f})'
    )

    streams, train_states = [], []
    for seed in seeds:
        rng = jax.random.key(seed)
        streams.append(ParallelMNISTStream(
            images=images, labels=labels, n_tasks=n_tasks,
            batch_size=cfg.train.batch_size, seed=seed,
            permute_period=permute_period,
            test_images=test_images, test_labels=test_labels,
        ))

        model = init_model(
            cfg, input_dim, output_dim, hidden_units, max_conns, max_fan_out,
            p_w1, p_w2,
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
            util_w1=jnp.zeros_like(model.network.weights),
            util_w2=jnp.zeros_like(model.network.output_weights),
            util_step_w1=jnp.zeros_like(model.network.weights, dtype=jnp.int32),
            util_step_w2=jnp.zeros_like(model.network.output_weights, dtype=jnp.int32),
        ))

    n_params = count_params(train_states[0].model)
    network0 = train_states[0].model.network
    n_active_conns = int(
        (network0.input_indices >= 0).sum() + network0.output_mask.sum()
    )
    initial_units = int(network0.unit_mask.sum())
    print(f'StructureModel: padded_params={n_params}, '
          f'active_connections={n_active_conns}, hidden_units={initial_units}, '
          f'seeds={seeds}')

    return stack_pytrees(train_states), streams, num_classes, n_tasks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig, train_state: TrainState, streams,
                   num_classes: int, n_tasks: int):
    log_freq = int(cfg.train.log_freq)
    num_log_periods = int(cfg.train.total_steps) // log_freq
    set_enabled = bool(cfg.set.enabled)
    evolve_freq = int(cfg.set.evolve_frequency) if set_enabled else log_freq
    eval_freq = int(cfg.train.get('eval_freq', 0))
    n_test_samples_cfg = cfg.train.get('n_test_samples', None)
    n_test_samples = int(n_test_samples_cfg) if n_test_samples_cfg is not None else None

    prune_metric = str(cfg.set.get('prune_metric', 'magnitude'))
    if prune_metric not in ('magnitude', 'utility'):
        raise ValueError(
            f"set.prune_metric must be 'magnitude' or 'utility', got {prune_metric!r}"
        )
    utility_decay = float(cfg.set.get('utility_decay', 0.999))
    print(f'SET pruning: metric={prune_metric}, utility_decay={utility_decay}')

    train_step_fn = partial(
        train_step, num_classes=num_classes, n_tasks=n_tasks,
        utility_decay=utility_decay,
    )

    if set_enabled:
        assert log_freq % evolve_freq == 0, (
            f'log_freq={log_freq} must be divisible by evolve_frequency={evolve_freq}'
        )
        evolve_cycles_per_log = log_freq // evolve_freq
        evolve_fn = partial(
            set_evolve, zeta=float(cfg.set.zeta),
            prune_metric=prune_metric, utility_decay=utility_decay,
        )

        def evolve_cycle(state, cycle_data):
            state, metrics = jax.lax.scan(train_step_fn, state, cycle_data, unroll=SCAN_UNROLL)
            state, n_p1, n_r1, n_p2, n_r2 = evolve_fn(state)
            return state, (metrics, n_p1 + n_p2, n_r1 + n_r2)

        def scan_log_period(state, data):
            data = jax.tree.map(
                lambda x: x.reshape(evolve_cycles_per_log, evolve_freq, *x.shape[1:]), data,
            )
            state, (metrics, pruned, regrown) = jax.lax.scan(evolve_cycle, state, data)
            return state, metrics.reshape(-1, *metrics.shape[2:]), pruned, regrown
    else:
        def scan_log_period(state, data):
            state, metrics = jax.lax.scan(train_step_fn, state, data, unroll=SCAN_UNROLL)
            return state, metrics, None, None

    vmapped_scan = jax.jit(jax.vmap(scan_log_period))

    all_losses, all_accs, all_per_seed_losses, all_per_seed_accs = [], [], [], []
    all_test_losses, all_test_accs = [], []
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    log_executor = ThreadPoolExecutor(max_workers=1)
    log_futures = []
    logging_active = (cfg.get('mlflow', False) or cfg.get('wandb', False)
                      or cfg.get('comet_ml', False))
    cumulative_pruned = 0.0
    cumulative_regrown = 0.0

    for _ in range(num_log_periods):
        batch = [s.sample_batch(log_freq) for s in streams]
        imgs = jnp.array(np.stack([b[0] for b in batch]))
        lbls = jnp.array(np.stack([b[1] for b in batch]))

        train_state, metrics, pruned, regrown = vmapped_scan(train_state, (imgs, lbls))

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
            cumulative_regrown += float(regrown.sum(axis=1).mean())
            structure_metrics['cumulative_pruned'] = cumulative_pruned
            structure_metrics['cumulative_regrown'] = cumulative_regrown

        test_metrics_dict = {}
        if eval_freq > 0 and step % eval_freq == 0:
            t_imgs, t_lbls = streams[0].get_test_batch()
            if n_test_samples is not None and n_test_samples < t_imgs.shape[0]:
                t_imgs = t_imgs[:n_test_samples]
                t_lbls = t_lbls[:n_test_samples]
            test_loss, test_acc = evaluate_test(
                train_state.model, t_imgs, t_lbls, num_classes, n_tasks,
            )
            mean_test_loss = float(test_loss.mean())
            mean_test_acc = float(test_acc.mean())
            all_test_losses.append(mean_test_loss)
            all_test_accs.append(mean_test_acc)
            test_metrics_dict = {
                'test_loss': mean_test_loss,
                'test_accuracy': mean_test_acc,
            }

        if logging_active:
            log_futures.append(log_executor.submit(
                _bg_log, mean_loss, std_loss, mean_acc, std_acc,
                per_seed_loss.tolist(), per_seed_acc.tolist(),
                structure_metrics, test_metrics_dict, cfg, step,
            ))

        all_losses.append(mean_loss)
        all_accs.append(mean_acc)
        all_per_seed_losses.append(np.array(per_seed_loss))
        all_per_seed_accs.append(np.array(per_seed_acc))

        pbar.update(log_freq)
        postfix = {
            'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}',
            'units': f'{structure_metrics["active_units"]:.0f}',
            'conn': f'{structure_metrics["active_connections"]:.0f}',
        }
        if test_metrics_dict:
            postfix['t_acc'] = f'{test_metrics_dict["test_accuracy"]:.4f}'
        pbar.set_postfix(postfix)

        if not np.isfinite(all_losses[-1]) or all_losses[-1] > DIVERGENCE_LOSS_THRESHOLD:
            print(f'\n[diverged] train loss {all_losses[-1]:.4g} exceeded threshold '
                  f'{DIVERGENCE_LOSS_THRESHOLD:.4g} at step {step}; stopping early.')
            break

    for f in log_futures:
        f.result()
    log_executor.shutdown(wait=False)
    pbar.close()
    return (train_state, all_losses, all_accs,
            all_per_seed_losses, all_per_seed_accs,
            all_test_losses, all_test_accs)


def _bg_log(mean_loss, std_loss, mean_acc, std_acc,
            per_seed_loss, per_seed_acc, structure_metrics,
            test_metrics_dict, cfg, step):
    metrics = {
        'loss': mean_loss, 'loss_std': std_loss,
        'accuracy': mean_acc, 'accuracy_std': std_acc,
    }
    metrics.update(structure_metrics)
    metrics.update(test_metrics_dict)
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
    (train_state, all_losses, all_accs, all_per_seed_losses, all_per_seed_accs,
     all_test_losses, all_test_accs) = run_experiment(
        cfg, train_state, streams, num_classes, n_tasks,
    )

    diverged = bool(
        all_losses
        and (not np.isfinite(all_losses[-1]) or all_losses[-1] > DIVERGENCE_LOSS_THRESHOLD)
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
        'diverged': float(diverged),
    }
    eval_freq = int(cfg.train.get('eval_freq', 0))
    if all_test_losses:
        n_test_tail = max(1, len(all_test_losses) // 10)
        summary['asymptotic_test_loss'] = float(np.mean(all_test_losses[-n_test_tail:]))
        summary['asymptotic_test_accuracy'] = float(np.mean(all_test_accs[-n_test_tail:]))
    elif eval_freq > 0:
        # Diverged before the first test eval -- emit NaN so downstream tools
        # (mlflow-sweeper's Optuna metric lookup) still find the key.
        summary['asymptotic_test_loss'] = float('nan')
        summary['asymptotic_test_accuracy'] = float('nan')
    print(f'Average loss: {summary["average_loss"]:.4f} | '
          f'Asymptotic loss: {summary["asymptotic_loss"]:.4f} | '
          f'Asymptotic acc: {summary["asymptotic_accuracy"]:.4f}')
    if all_test_losses:
        print(f'Asymptotic test loss: {summary["asymptotic_test_loss"]:.4f} | '
              f'Asymptotic test acc: {summary["asymptotic_test_accuracy"]:.4f}')
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


@hydra.main(config_path='../conf', config_name='train_set', version_base='1.1')
def main(cfg: DictConfig) -> None:
    cfg = init_experiment(cfg.project, cfg)
    try:
        run_config(cfg)
    finally:
        finish_experiment(cfg)


if __name__ == '__main__':
    main()
