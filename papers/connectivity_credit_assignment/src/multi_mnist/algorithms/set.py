"""Sparse Evolutionary Training (SET), Mocanu et al., Nature Comms 2018, on ``PaddedMLP``.

    1. Initialize each bipartite layer (W1: input->hidden, W2: hidden->output)
       as an Erdos-Renyi random graph with edge probability
       p(W_ij) = epsilon * (n_in + n_out) / (n_in * n_out). That is
       ``model.init_strategy: sparse``.
    2. Every ``evolve_frequency`` steps, per layer: remove a fraction zeta of
       the smallest-positive active weights and a fraction zeta of the
       largest-negative ones (the negatives closest to zero), then regrow the
       same number at random inactive positions.
    3. Hidden units stay permanently active; only connections evolve.

Paper defaults: epsilon=20, zeta=0.3 -- and note the paper's unit of time is an
**epoch**. Algorithm 1 line 9 is "for each training epoch", so zeta=0.3 rewires
30% of a layer per pass over the training set, about 5e-6 of it per example on
MNIST. This repository trains online at batch size 1 with no epoch to key off,
so the sweeps grid the per-example turnover rate directly and derive zeta from
it; see experiments/sweeps/04_connectivity_methods/sweep/set.yaml.

Representation
--------------
Connectivity is a mask over a dense matrix, so both the removal and the
regrowth are **global within a layer**: the layer's connection count is
preserved, but nothing holds an individual unit's fan-in fixed, which is what
lets the degree distribution drift away from its binomial initialization as
the paper describes. ``algorithms/dynamic/set.py`` is the same algorithm on
the preallocated sparse representation, where storage is per-unit slots and
regrowth is therefore confined to the row it was removed from; use that one
when the network is too large to pay the dense matmul.

This is also the worked reference for the algorithm interface. Note what it
needs that the static baselines do not: a traced periodic event, per-weight
optimizer-state resets when a connection is regrown, and -- for the utility
variant -- per-step state.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ..models.padded_mlp import PaddedMLP
from ..utils import tree_replace
from ._padded_optim import reset_optimizer_at
from ._selection import (
    bias_corrected, random_inactive_mask, signed_prune_mask,
    smallest_score_prune_mask,
)
from .base import ConnectivityAlgorithm


class SETState(eqx.Module):
    """Contribution-utility traces and their per-weight step counts.

    Only built when ``prune_metric == 'utility'``; the state is ``None`` for
    published SET, so the magnitude path allocates nothing.
    """
    util_w1: jax.Array
    util_w2: jax.Array
    step_w1: jax.Array
    step_w2: jax.Array


def _evolve_layer(weights, mask, eligible, score, zeta, key, *, prune_metric):
    """One prune-and-regrow pass over a layer."""
    active = mask.astype(jnp.bool_)

    if prune_metric == 'utility':
        prune = smallest_score_prune_mask(score, active, zeta)
    else:
        prune = signed_prune_mask(weights, active, zeta)

    after_prune = active & ~prune
    n_prune = prune.sum().astype(jnp.int32)

    # Global regrowth: a replacement may land anywhere in the layer, subject
    # only to the receiving unit slot being in use.
    regrow = random_inactive_mask(after_prune | ~eligible, n_prune, key)
    new_active = after_prune | regrow

    # Regrown connections enter at exactly 0.
    #
    # The paper does not say: Algorithm 1 line 16 is "add randomly new weights
    # (connections) in the same amount as the ones removed previously", where
    # "randomly" modifies the position, not the value. The authors' two released
    # implementations disagree -- the Keras one enters at 0 (the new mask is
    # applied as a constraint, so the position must be grown by gradient), the
    # sparse-data-structures one at N(0, 0.1^2). This follows the Keras one.
    #
    # The consequence is worth knowing: `signed_prune_mask` excludes weights of
    # exactly 0 from both the positive and negative sets, so a new connection is
    # unprunable until the first gradient moves it off zero, and is then the
    # smallest weight in the layer. It therefore gets one event's worth of
    # gradient steps to establish itself or be removed again -- which is why the
    # evolution rate, not just zeta, decides whether anything can grow.
    new_weights = jnp.where(regrow, 0.0, jnp.where(prune, 0.0, weights))
    return (new_weights, new_active.astype(mask.dtype), prune | regrow,
            n_prune, regrow.sum().astype(jnp.int32))


class SET(ConnectivityAlgorithm):
    """SET as a connectivity algorithm.

    ``zeta = 0`` freezes the topology at its Erdos-Renyi initialization, but
    prefer ``algorithm.name: static`` for that control -- it skips the event
    entirely instead of running it to no effect.
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
                f"set.prune_metric must be 'magnitude' or 'utility', "
                f"got {prune_metric!r}")
        self.event_period = int(evolve_frequency)
        self.zeta = float(zeta)
        self.prune_metric = prune_metric
        self.utility_decay = float(utility_decay)

    def init_state(self, model: PaddedMLP, *, key: PRNGKeyArray):
        if self.prune_metric != 'utility':
            return None
        return SETState(
            util_w1=jnp.zeros_like(model.W1),
            util_w2=jnp.zeros_like(model.W2),
            step_w1=jnp.zeros_like(model.W1, dtype=jnp.int32),
            step_w2=jnp.zeros_like(model.W2, dtype=jnp.int32),
        )

    def step_hook(self, algo_state, model_pre, model_post, aux):
        """EMA of ``|source activation| * |weight|`` on the pre-update weights.

        Pairing post-update weights with the activations the *pre-update*
        weights produced would mismatch the two halves of the product, so this
        deliberately reads ``model_pre``.
        """
        if self.prune_metric != 'utility':
            return algo_state

        images, hidden = aux
        decay = self.utility_decay

        # Reduce over the batch before the outer product: abs and mean both
        # commute with it, and it keeps the intermediate at weight-matrix size.
        x_mag = jnp.mean(jnp.abs(images), axis=0)                 # (input_dim,)
        h_mag = jnp.mean(jnp.abs(hidden), axis=0)                 # (max_hidden,)

        contrib_w1 = x_mag[None, :] * jnp.abs(model_pre.W1)
        contrib_w2 = h_mag[None, :] * jnp.abs(model_pre.W2)

        return SETState(
            util_w1=decay * algo_state.util_w1 + (1.0 - decay) * contrib_w1,
            util_w2=decay * algo_state.util_w2 + (1.0 - decay) * contrib_w2,
            step_w1=algo_state.step_w1 + 1,
            step_w2=algo_state.step_w2 + 1,
        )

    def event(self, model: PaddedMLP, optimizer, algo_state, *, key):
        k1, k2 = jax.random.split(key)

        if self.prune_metric == 'utility':
            score_w1 = bias_corrected(
                algo_state.util_w1, algo_state.step_w1, self.utility_decay)
            score_w2 = bias_corrected(
                algo_state.util_w2, algo_state.step_w2, self.utility_decay)
        else:
            score_w1 = score_w2 = None

        # Only slots belonging to an active hidden unit are eligible.
        W1, w1_mask, touched_w1, p1, r1 = _evolve_layer(
            model.W1, model.w1_mask, model.unit_mask[:, None] > 0,
            score_w1, self.zeta, k1, prune_metric=self.prune_metric)
        W2, w2_mask, touched_w2, p2, r2 = _evolve_layer(
            model.W2, model.w2_mask, model.unit_mask[None, :] > 0,
            score_w2, self.zeta, k2, prune_metric=self.prune_metric)

        new_model = tree_replace(
            model, W1=W1, W2=W2, w1_mask=w1_mask, w2_mask=w2_mask)
        new_optimizer = reset_optimizer_at(optimizer, touched_w1, touched_w2)

        new_state = algo_state
        if self.prune_metric == 'utility':
            # A rewired slot holds a new connection, so its trace restarts.
            new_state = SETState(
                util_w1=jnp.where(touched_w1, 0.0, algo_state.util_w1),
                util_w2=jnp.where(touched_w2, 0.0, algo_state.util_w2),
                step_w1=jnp.where(touched_w1, 0, algo_state.step_w1),
                step_w2=jnp.where(touched_w2, 0, algo_state.step_w2),
            )

        info = {'pruned': p1 + p2, 'regrown': r1 + r2}
        return new_model, new_optimizer, new_state, info
