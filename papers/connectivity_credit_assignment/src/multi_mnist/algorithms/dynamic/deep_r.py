"""DEEP-R (Bellec et al., ICLR 2018), on ``DynamicNetwork``.

DEEP-R performs Bayesian sampling over network structure. Each connection
carries a fixed sign; training does constrained SGD on the magnitude with an
L1 pull toward zero and Langevin noise::

    w <- w - lr * dL/dw - lr * l1 * sign(w) + sqrt(2 * lr * T) * nu

When a weight crosses zero the connection is deactivated, and a dormant
connection is reactivated at random to keep the connection count constant.
The noise term is what makes it a sampler rather than a pruner: connections
keep being tried, so the network explores topologies instead of committing to
an early one.

Split across two rates
----------------------
The two halves of that description cost wildly different amounts. The weight
update is elementwise and cheap. The rewiring is not: reactivating a
connection needs a random draw over every dormant slot in the row, and any
change to the topology forces ``build_outgoing_indices`` to rebuild the
backward pass's index tables with a sort over every connection slot. Run every
step, the rewiring costs ~8x the rest of training.

So the two run at different periods:

``step_update`` (every step)
    Apply the L1 pull and the noise, and *flag* any connection whose weight
    crossed zero -- zeroing its weight and clearing its index. A flagged
    connection is immediately dormant in exactly the paper's sense: it
    contributes nothing to the forward pass and receives no gradient.

``event`` (every ``event_period`` steps)
    Reactivate enough random dormant connections to restore each row to its
    initial count, then rebuild the index tables.

The weight dynamics are therefore *identical* to the published algorithm at
any ``event_period``. The single deviation is that the active connection
count dips between events instead of being restored instantly, recovering at
each event; the average shortfall is about ``event_period / 2`` times the
per-step flip rate, which the logged ``active_connections`` makes visible.

Note that Algorithm 1 in the paper rewires once per *iteration*, and an
iteration there is a minibatch update. At ``batch_size: 1`` an
``event_period`` in the tens is closer to the paper's cadence measured in
samples than ``event_period: 1`` is.

Two further deviations from the paper, both forced by the representation and
both shared with this repository's SET implementation:

1. **Connection count is preserved per hidden unit, not globally.**
   ``DynamicNetwork`` stores connections as a fixed number of slots per unit,
   so a regrown connection has to land in the row it was pruned from. Global
   rebalancing would need a different storage layout.
2. **The L1 and noise terms are applied after the optimizer step** rather than
   folded into it. For SGD these are identical; for a preconditioned optimizer
   they are not, so treat ``optimizer.name: sgd`` as the supported setting.
"""

from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ...models.dynamic_network import build_outgoing_indices
from ...optim import EqxOptimizer
from ...utils import tree_replace
from ..base import ConnectivityAlgorithm
from ...models.sparse_init import HIDDEN_LAYER, StructureModel
from .set import _reset_optimizer_at


class DeepRState(eqx.Module):
    """Per-row connection counts, measured once at initialization.

    The event restores each row to these counts. Deriving the target from the
    initial topology rather than from the current one is what lets pruning and
    regrowth happen at different times without the budget drifting.
    """
    target_fan_in_w1: jax.Array   # (units,)
    target_fan_in_w2: jax.Array   # (outputs,)


def _langevin_update(weights, active, lr, l1, temperature, key):
    """Apply the L1 pull and Langevin noise to the active weights."""
    noise_scale = jnp.sqrt(2.0 * lr * temperature)
    noise = jax.random.normal(key, weights.shape) * noise_scale
    updated = weights - lr * l1 * jnp.sign(weights) + noise
    return jnp.where(active, updated, weights)


# ---------------------------------------------------------------------------
# Per step: noise, L1, and flagging zero-crossings as dormant
# ---------------------------------------------------------------------------

def _flag_w1(network, key, *, lr, l1, temperature):
    weights = network.weights[HIDDEN_LAYER]          # (U, C)
    idx = network.input_indices[HIDDEN_LAYER]        # (U, C)
    active = idx >= 0

    sign_before = jnp.sign(weights)
    weights_noisy = _langevin_update(weights, active, lr, l1, temperature, key)

    # A connection dies when its weight crosses zero. Zeroing the weight is
    # what makes it dormant right away: the forward pass reads this slot's
    # weight, and sync_outgoing_weights mirrors the zero into the backward
    # pass, so the stale outgoing index left until the next event is inert.
    flipped = active & (jnp.sign(weights_noisy) != sign_before)
    weights_after = jnp.where(flipped, 0.0, weights_noisy)
    idx_after = jnp.where(flipped, jnp.int32(-1), idx)

    return tree_replace(
        network,
        weights=network.weights.at[HIDDEN_LAYER].set(weights_after),
        input_indices=network.input_indices.at[HIDDEN_LAYER].set(idx_after),
    )


def _flag_w2(network, key, *, lr, l1, temperature):
    input_dim = network.input_dim
    s, e = input_dim, input_dim + network.max_units_per_layer

    weights = network.output_weights[:, s:e]                 # (O, U)
    active = network.output_mask[:, s:e].astype(jnp.bool_)

    sign_before = jnp.sign(weights)
    weights_noisy = _langevin_update(weights, active, lr, l1, temperature, key)

    flipped = active & (jnp.sign(weights_noisy) != sign_before)
    weights_after = jnp.where(flipped, 0.0, weights_noisy)
    active_after = active & ~flipped

    return tree_replace(
        network,
        output_weights=network.output_weights.at[:, s:e].set(weights_after),
        output_mask=network.output_mask.at[:, s:e].set(
            active_after.astype(network.output_mask.dtype)),
    )


# ---------------------------------------------------------------------------
# Every event_period steps: reactivate dormant connections, rebuild indices
# ---------------------------------------------------------------------------

def _regrow_w1(network, optimizer, target_fan_in, key):
    """Restore each unit's incoming count by activating random dormant slots."""
    weights = network.weights[HIDDEN_LAYER]
    idx = network.input_indices[HIDDEN_LAYER]
    active = idx >= 0
    U, C = weights.shape
    input_dim = network.input_dim

    deficit = target_fan_in - active.sum(axis=-1)      # (U,)
    row_keys = jax.random.split(key, U)

    def regrow_row(row_idx, row_w, n_to_regrow, key):
        active_slot = row_idx >= 0
        safe_col = jnp.where(active_slot, row_idx, 0)
        in_use = jnp.zeros(input_dim, dtype=jnp.bool_).at[safe_col].set(active_slot)

        col_score = jnp.where(
            in_use, -jnp.inf, jax.random.uniform(key, (input_dim,)))
        sort_order = jnp.argsort(-col_score)

        free_slot = ~active_slot
        free_rank = jnp.cumsum(free_slot.astype(jnp.int32)) - 1
        slot_takes = free_slot & (free_rank >= 0) & (free_rank < n_to_regrow)
        chosen_col = sort_order[jnp.clip(free_rank, 0, input_dim - 1)]
        new_row_idx = jnp.where(slot_takes, chosen_col.astype(jnp.int32), row_idx)

        # Reactivated connections start at zero, as DEEP-R prescribes: they
        # re-enter at the boundary they were pruned at, not at a random large
        # value that would perturb the function.
        new_row_w = jnp.where(slot_takes, 0.0, row_w)
        return new_row_idx, new_row_w, slot_takes

    new_idx, new_w, regrow_mask = jax.vmap(regrow_row)(
        idx, weights, deficit, row_keys)

    new_network = tree_replace(
        network,
        input_indices=network.input_indices.at[HIDDEN_LAYER].set(new_idx),
        weights=network.weights.at[HIDDEN_LAYER].set(new_w),
    )
    # Reset optimizer state on every slot that is not a continuously-active
    # connection: the ones regrown now, plus those flagged since the last
    # event. Padding slots are swept up too and are always zero anyway.
    reset = jnp.zeros_like(network.weights, dtype=bool).at[HIDDEN_LAYER].set(
        regrow_mask | ~active)
    new_optimizer = _reset_optimizer_at(
        optimizer, reset, jnp.zeros_like(network.output_weights, dtype=bool))

    return new_network, new_optimizer, regrow_mask.sum().astype(jnp.int32)


def _regrow_w2(network, optimizer, target_fan_in, key):
    """Restore each output's incoming count by activating random dormant slots."""
    input_dim = network.input_dim
    U = network.max_units_per_layer
    s, e = input_dim, input_dim + U

    weights = network.output_weights[:, s:e]
    active = network.output_mask[:, s:e].astype(jnp.bool_)
    O = weights.shape[0]

    deficit = target_fan_in - active.sum(axis=-1)
    row_keys = jax.random.split(key, O)

    def regrow_row(row_active, row_w, n_to_regrow, key):
        col_score = jnp.where(
            row_active, -jnp.inf, jax.random.uniform(key, (U,)))
        rank = jnp.argsort(jnp.argsort(-col_score))
        slot_takes = (rank < n_to_regrow) & ~row_active
        new_row_w = jnp.where(slot_takes, 0.0, row_w)
        return new_row_w, row_active | slot_takes, slot_takes

    new_w, new_active, regrow_mask = jax.vmap(regrow_row)(
        active, weights, deficit, row_keys)

    new_network = tree_replace(
        network,
        output_weights=network.output_weights.at[:, s:e].set(new_w),
        output_mask=network.output_mask.at[:, s:e].set(
            new_active.astype(network.output_mask.dtype)),
    )
    ow_reset = jnp.zeros_like(network.output_weights, dtype=bool).at[:, s:e].set(
        regrow_mask | ~active)
    new_optimizer = _reset_optimizer_at(
        optimizer, jnp.zeros_like(network.weights, dtype=bool), ow_reset)

    return new_network, new_optimizer, regrow_mask.sum().astype(jnp.int32)


class DeepR(ConnectivityAlgorithm):
    """DEEP-R as a connectivity algorithm."""

    name = 'deep_r'
    needs_step_key = True

    def __init__(
        self,
        learning_rate: float,
        l1: float = 1e-3,
        temperature: float = 1e-5,
        evolve_w2: bool = True,
        event_period: int = 25,
    ):
        """
        Args:
            learning_rate: Must match ``optimizer.learning_rate``; it scales
                both the L1 pull and the noise, so the two cannot drift apart.
            l1: L1 coefficient (``alpha`` in the paper). It has to be strong
                enough to move a weight on the problem's own timescale: the
                steps needed to walk a typical weight to zero are
                ``|w| / (lr * l1)``.
            temperature: Langevin temperature ``T``. The noise displaces a
                weight by ``sqrt(N * 2 * lr * T)`` over ``N`` steps. Setting it
                to 0 turns DEEP-R into deterministic sign-flip pruning with
                random regrowth, which is a useful ablation.
            evolve_w2: Whether the output layer also evolves. False keeps the
                hidden->output connectivity fixed and evolves only the input
                layer.
            event_period: Steps between reactivation events. Does **not**
                change the weight dynamics -- pruning is still detected every
                step -- only how long a connection stays dormant before being
                replaced. See the module docstring.
        """
        self.learning_rate = float(learning_rate)
        self.l1 = float(l1)
        self.temperature = float(temperature)
        self.evolve_w2 = bool(evolve_w2)
        self.event_period = int(event_period)

    def init_state(self, model: StructureModel, *, key: PRNGKeyArray) -> DeepRState:
        network = model.network
        s = network.input_dim
        e = s + network.max_units_per_layer
        return DeepRState(
            target_fan_in_w1=(
                network.input_indices[HIDDEN_LAYER] >= 0).sum(axis=-1).astype(jnp.int32),
            target_fan_in_w2=(
                network.output_mask[:, s:e].astype(jnp.bool_)).sum(axis=-1).astype(jnp.int32),
        )

    def step_update(self, model, algo_state, *, key):
        k1, k2 = jax.random.split(key)
        network = _flag_w1(
            model.network, k1,
            lr=self.learning_rate, l1=self.l1, temperature=self.temperature)
        if self.evolve_w2:
            network = _flag_w2(
                network, k2,
                lr=self.learning_rate, l1=self.l1, temperature=self.temperature)
        return tree_replace(model, network=network), algo_state

    def event(self, model, optimizer, algo_state, *, key):
        k1, k2 = jax.random.split(key)
        network = model.network

        network, optimizer, r1 = _regrow_w1(
            network, optimizer, algo_state.target_fan_in_w1, k1)

        if self.evolve_w2:
            network, optimizer, r2 = _regrow_w2(
                network, optimizer, algo_state.target_fan_in_w2, k2)
        else:
            r2 = jnp.int32(0)

        # The topology changed, so the outgoing index table and its mirrored
        # weights have to be rebuilt before the next backward pass.
        network = build_outgoing_indices(network)

        # Every regrown connection replaces one flagged since the last event,
        # so the two counts are equal by construction.
        info = {'pruned': r1 + r2, 'regrown': r1 + r2}
        return tree_replace(model, network=network), optimizer, algo_state, info
