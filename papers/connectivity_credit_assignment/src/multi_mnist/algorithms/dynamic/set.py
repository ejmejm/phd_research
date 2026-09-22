"""Sparse Evolutionary Training (SET), Mocanu et al., Nature Comms 2018.

    1. Initialize each bipartite layer (W1: input->hidden, W2: hidden->output)
       as an Erdos-Renyi random graph with edge probability
       p(W_ij) = epsilon * (n_in + n_out) / (n_in * n_out).
    2. Every ``evolve_frequency`` steps, per layer: remove a fraction zeta of
       the smallest-positive active weights and a fraction zeta of the
       largest-negative ones (the negatives closest to zero), then regrow the
       same number at random inactive positions in the same row -- so per-unit
       fan-in, and the layer's connection count, are exactly preserved.
    3. Hidden units stay permanently active; only connections evolve.

Paper defaults: epsilon=20, zeta=0.3.

This is the worked reference for the algorithm interface. Note what it needs
that the static baselines do not: per-step state (the utility traces), a
traced periodic event, and per-weight optimizer-state resets when a
connection is regrown. A new method that needs those has a template here.

SET runs on ``DynamicNetwork`` rather than ``PaddedMLP`` because it is only
worth running at sizes where a masked dense matmul is wasteful.
"""

from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig

from ...models.dynamic_network import DynamicNetwork, build_outgoing_indices
from ...models.sparse_init import HIDDEN_LAYER, StructureModel
from ...optim import AdamState, EqxOptimizer
from ...utils import tree_replace
from .._selection import (
    bias_corrected as _bias_corrected_utility,
    signed_prune_mask as _signed_prune_threshold,
    smallest_score_prune_mask as _utility_prune_threshold,
)
from ..base import ConnectivityAlgorithm


class SETState(eqx.Module):
    """Contribution-utility traces and their per-weight step counts.

    Always maintained, but only read when ``prune_metric == 'utility'``. With
    the default ``magnitude`` metric the behaviour is bit-equivalent to SET as
    published.
    """
    util_w1: jax.Array
    util_w2: jax.Array
    util_step_w1: jax.Array
    util_step_w2: jax.Array


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


def set_evolve(
    model: StructureModel, optimizer: EqxOptimizer, algo_state: SETState,
    key: PRNGKeyArray, *, zeta: float, prune_metric: str, utility_decay: float,
):
    """One SET prune-and-regrow event over both bipartite layers."""
    k_w1, k_w2 = jax.random.split(key, 2)

    network, optimizer, util_w1, util_step_w1, n_p1, n_r1 = _set_evolve_w1(
        model.network, optimizer, algo_state.util_w1, algo_state.util_step_w1,
        k_w1, zeta,
        prune_metric=prune_metric, utility_decay=utility_decay,
    )
    network, optimizer, util_w2, util_step_w2, n_p2, n_r2 = _set_evolve_w2(
        network, optimizer, algo_state.util_w2, algo_state.util_step_w2,
        k_w2, zeta,
        prune_metric=prune_metric, utility_decay=utility_decay,
    )
    # Pruning and regrowing changed which units feed which, so the outgoing
    # index table has to be rebuilt before the next backward pass.
    network = build_outgoing_indices(network)

    new_model = tree_replace(model, network=network)
    new_algo_state = SETState(
        util_w1=util_w1, util_w2=util_w2,
        util_step_w1=util_step_w1, util_step_w2=util_step_w2,
    )
    info = {'pruned': n_p1 + n_p2, 'regrown': n_r1 + n_r2}
    return new_model, optimizer, new_algo_state, info


def _w1_source_magnitude(images, input_indices):
    """Batch-mean ``|x_i|`` for each W1 slot, as a (units, conns) array.

    ``input_indices[u, c] = i`` is the input index feeding slot (u, c), or -1
    for an inactive slot (which contributes 0).

    The batch is reduced *before* the per-slot gather. That is exact -- abs and
    mean both commute with a gather, so every slot still receives the mean of
    the same B values -- and it keeps the intermediate at (units, conns)
    instead of (B, units, conns). Only the trace is affected; the sparse
    forward pass gathers per-slot inputs itself and is unavoidably B-sized.
    """
    active = input_indices >= 0
    safe_idx = jnp.where(active, input_indices, 0)
    x_abs_mean = jnp.mean(jnp.abs(images), axis=0)                      # (I,)
    return jnp.where(active, x_abs_mean[safe_idx], 0.0)                # (U, C)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class SET(ConnectivityAlgorithm):
    """SET as a connectivity algorithm.

    ``zeta = 0`` freezes the topology at its Erdos-Renyi initialization, which
    is the natural static random-sparse control: same connection count, same
    degree distribution, no evolution.
    """

    name = 'set'

    def __init__(
        self,
        evolve_frequency: int,
        zeta: float,
        prune_metric: str = 'magnitude',
        utility_decay: float = 0.999,
    ):
        if prune_metric not in ('magnitude', 'utility'):
            raise ValueError(
                f"set.prune_metric must be 'magnitude' or 'utility', got {prune_metric!r}")
        self.event_period = int(evolve_frequency)
        self.zeta = float(zeta)
        self.prune_metric = prune_metric
        self.utility_decay = float(utility_decay)

    def init_state(self, model: StructureModel, *, key: PRNGKeyArray) -> SETState:
        network = model.network
        return SETState(
            util_w1=jnp.zeros_like(network.weights),
            util_w2=jnp.zeros_like(network.output_weights),
            util_step_w1=jnp.zeros_like(network.weights, dtype=jnp.int32),
            util_step_w2=jnp.zeros_like(network.output_weights, dtype=jnp.int32),
        )

    def step_hook(self, algo_state, model_pre, model_post, aux):
        """EMA of |source activation| * |weight|, on the pre-update weights.

        Pairing post-update weights with the activations the *pre-update*
        weights produced would mismatch the two halves of the product, so this
        deliberately reads ``model_pre``.
        """
        images, buffer = aux
        network = model_pre.network
        decay = self.utility_decay
        input_dim = network.input_dim
        max_units = network.max_units_per_layer

        # W1: |x_i| * |w_{u,c}| where i = input_indices[u, c].
        x_mag = _w1_source_magnitude(images, network.input_indices[HIDDEN_LAYER])
        contrib_w1 = x_mag * jnp.abs(network.weights[HIDDEN_LAYER])
        util_w1 = algo_state.util_w1.at[HIDDEN_LAYER].set(
            decay * algo_state.util_w1[HIDDEN_LAYER] + (1.0 - decay) * contrib_w1)
        util_step_w1 = algo_state.util_step_w1.at[HIDDEN_LAYER].set(
            algo_state.util_step_w1[HIDDEN_LAYER] + 1)

        # W2: |h_j| * |w_{k,j}|. The hidden activations sit in the value
        # buffer at [input_dim : input_dim + max_units].
        s, e = input_dim, input_dim + max_units
        h_mag = jnp.mean(jnp.abs(buffer[:, s:e]), axis=0)
        contrib_w2 = h_mag[None, :] * jnp.abs(network.output_weights[:, s:e])
        util_w2 = algo_state.util_w2.at[:, s:e].set(
            decay * algo_state.util_w2[:, s:e] + (1.0 - decay) * contrib_w2)
        util_step_w2 = algo_state.util_step_w2.at[:, s:e].set(
            algo_state.util_step_w2[:, s:e] + 1)

        return SETState(
            util_w1=util_w1, util_w2=util_w2,
            util_step_w1=util_step_w1, util_step_w2=util_step_w2,
        )

    def event(self, model, optimizer, algo_state, *, key):
        return set_evolve(
            model, optimizer, algo_state, key,
            zeta=self.zeta, prune_metric=self.prune_metric,
            utility_decay=self.utility_decay,
        )
