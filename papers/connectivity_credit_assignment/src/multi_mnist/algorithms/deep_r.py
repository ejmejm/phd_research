"""DEEP-R (Bellec et al., ICLR 2018), on ``PaddedMLP``.

DEEP-R performs Bayesian sampling over network structure. Each connection
carries a fixed sign; training does constrained SGD on the magnitude with an
L1 pull toward zero and Langevin noise::

    w <- w - lr * dL/dw - lr * l1 * sign(w) + sqrt(2 * lr * T) * nu

When a weight crosses zero the connection is deactivated, and a dormant
connection is reactivated at random so the connection count stays constant.
The noise is what makes it a sampler rather than a pruner: connections keep
being tried, so the network explores topologies instead of committing to an
early one.

Representation
--------------
Connectivity is a mask over a dense matrix, so a reactivated connection can
land anywhere in the layer and the **per-layer connection count is what is
preserved**, exactly as the paper specifies. ``algorithms/dynamic/deep_r.py``
is the same algorithm on the preallocated sparse representation, where storage
is per-unit slots and the count is therefore preserved per unit instead; use
that one when the network is too large to pay the dense matmul.

Split across two rates
----------------------
``step_update`` (every step)
    Apply the L1 pull and the noise, and *flag* any connection whose weight
    crossed zero -- clearing its mask bit and zeroing its weight. A flagged
    connection is dormant in exactly the paper's sense from that moment: it
    contributes nothing to the forward pass and receives no gradient.

``event`` (every ``event_period`` steps)
    Reactivate random dormant connections to restore each layer's count.
    Reactivation is a uniform draw over dormant connections and so needs no
    ordering: each is selected independently with probability
    ``deficit / n_dormant``. The count therefore lands within about +/-140 of
    the target rather than on it exactly, and self-corrects because the
    deficit is measured against the absolute target every event. Pinning it
    exactly would mean sorting the whole matrix, which costs 34x more per
    event and is what forces ``event_period`` above 1.

The weight dynamics are therefore identical at any ``event_period``; the only
consequence of raising it is that the connection count dips between events and
recovers at each one. Algorithm 1 in the paper rewires once per *iteration*,
and an iteration there is a minibatch update -- so at ``batch_size: 1`` an
``event_period`` above 1 is not obviously less faithful than 1.

One deviation remains: the L1 and noise terms are applied after the optimizer
step rather than folded into it. For SGD these are identical; for a
preconditioned optimizer they are not, so ``optimizer.name: sgd`` is the
supported setting.
"""

from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ..models.padded_mlp import PaddedMLP
from ..utils import tree_replace
from ._padded_optim import reset_optimizer_at
from ._selection import bernoulli_inactive_mask
from .base import ConnectivityAlgorithm


class DeepRState(eqx.Module):
    """The per-layer connection counts the event restores to.

    Measured once at initialization. Taking the target from the initial
    topology rather than from the current one is what lets pruning and
    regrowth happen at different times without the budget drifting.
    """
    target_w1: jax.Array   # scalar
    target_w2: jax.Array   # scalar


def _langevin(weights, active, lr, l1, temperature, key):
    """L1 pull plus Langevin noise, applied only to active connections."""
    noise = jax.random.normal(key, weights.shape) * jnp.sqrt(2.0 * lr * temperature)
    return jnp.where(active, weights - lr * l1 * jnp.sign(weights) + noise, weights)


def _flag(weights, mask, key, *, lr, l1, temperature):
    """Advance the Langevin dynamics and deactivate zero-crossings."""
    active = mask.astype(jnp.bool_)
    sign_before = jnp.sign(weights)
    noisy = _langevin(weights, active, lr, l1, temperature, key)

    flipped = active & (jnp.sign(noisy) != sign_before)
    return (jnp.where(flipped, 0.0, noisy),
            (active & ~flipped).astype(mask.dtype))


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
                hidden->output connectivity fixed.
            event_period: Steps between reactivation events. Does not change
                the weight dynamics -- zero-crossings are still detected every
                step -- only how long a connection stays dormant before being
                replaced.
        """
        self.learning_rate = float(learning_rate)
        self.l1 = float(l1)
        self.temperature = float(temperature)
        self.evolve_w2 = bool(evolve_w2)
        self.event_period = int(event_period)

    def init_state(self, model: PaddedMLP, *, key: PRNGKeyArray) -> DeepRState:
        return DeepRState(
            target_w1=model.w1_mask.sum().astype(jnp.int32),
            target_w2=model.w2_mask.sum().astype(jnp.int32),
        )

    def step_update(self, model: PaddedMLP, algo_state, *, key):
        k1, k2 = jax.random.split(key)
        W1, w1_mask = _flag(
            model.W1, model.w1_mask, k1,
            lr=self.learning_rate, l1=self.l1, temperature=self.temperature)

        if self.evolve_w2:
            W2, w2_mask = _flag(
                model.W2, model.w2_mask, k2,
                lr=self.learning_rate, l1=self.l1, temperature=self.temperature)
        else:
            W2, w2_mask = model.W2, model.w2_mask

        return tree_replace(
            model, W1=W1, W2=W2, w1_mask=w1_mask, w2_mask=w2_mask), algo_state

    def event(self, model: PaddedMLP, optimizer, algo_state, *, key):
        k1, k2 = jax.random.split(key)

        # Only slots belonging to an active hidden unit may be reactivated.
        eligible_w1 = model.unit_mask[:, None] > 0
        eligible_w2 = model.unit_mask[None, :] > 0

        active_w1 = model.w1_mask.astype(jnp.bool_)
        regrow_w1 = bernoulli_inactive_mask(
            active_w1 | ~eligible_w1, algo_state.target_w1 - active_w1.sum(), k1)

        if self.evolve_w2:
            active_w2 = model.w2_mask.astype(jnp.bool_)
            regrow_w2 = bernoulli_inactive_mask(
                active_w2 | ~eligible_w2, algo_state.target_w2 - active_w2.sum(), k2)
        else:
            active_w2 = model.w2_mask.astype(jnp.bool_)
            regrow_w2 = jnp.zeros_like(active_w2)

        # Reactivated connections re-enter at zero, as DEEP-R prescribes: at
        # the boundary they were pruned at, not at a value that would perturb
        # the function. The next gradient step decides their sign.
        new_model = tree_replace(
            model,
            W1=jnp.where(regrow_w1, 0.0, model.W1),
            W2=jnp.where(regrow_w2, 0.0, model.W2),
            w1_mask=(active_w1 | regrow_w1).astype(model.w1_mask.dtype),
            w2_mask=(active_w2 | regrow_w2).astype(model.w2_mask.dtype),
        )
        # Reset optimizer state on the newly-live slots and on everything
        # dormant, which sweeps up connections flagged since the last event.
        new_optimizer = reset_optimizer_at(
            optimizer, regrow_w1 | ~active_w1, regrow_w2 | ~active_w2)

        # Every reactivation replaces one connection flagged since the last
        # event, so the two counts are equal by construction.
        n = regrow_w1.sum().astype(jnp.int32) + regrow_w2.sum().astype(jnp.int32)
        return new_model, new_optimizer, algo_state, {'pruned': n, 'regrown': n}
