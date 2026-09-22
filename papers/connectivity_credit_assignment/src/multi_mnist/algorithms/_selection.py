"""Choosing which connections to remove and where to put new ones.

Shared by SET and DEEP-R, and by both the ``PaddedMLP`` and ``DynamicNetwork``
implementations of each, so that the selection rule is defined once.

Every function here returns a boolean mask of the same shape as its input.
Nothing changes shape, so all of it is safe inside the jitted, vmapped scan.
"""

import jax
import jax.numpy as jnp


def signed_prune_mask(weights, active, zeta):
    """SET's rule: the smallest-positive and largest-negative fractions.

    ``zeta`` of the active positive weights closest to zero and ``zeta`` of the
    active negative weights closest to zero are selected, by sign separately.
    Thresholds come from a sort over the whole array, so pruning is global
    within the layer -- which is what the SET paper specifies.
    """
    flat = weights.reshape(-1)
    flat_active = active.reshape(-1)

    n_pos = jnp.sum(flat_active & (flat > 0))
    n_neg = jnp.sum(flat_active & (flat < 0))
    n_prune_pos = (zeta * n_pos.astype(jnp.float32)).astype(jnp.int32)
    n_prune_neg = (zeta * n_neg.astype(jnp.float32)).astype(jnp.int32)

    pos_score = jnp.where(flat_active & (flat > 0), flat, jnp.inf)
    threshold_pos = jnp.sort(pos_score)[jnp.maximum(n_prune_pos - 1, 0)]
    prune_pos = (flat_active & (flat > 0) & (flat <= threshold_pos)
                 & (n_prune_pos > 0))

    # "Largest negative" means closest to zero, i.e. smallest -w.
    neg_score = jnp.where(flat_active & (flat < 0), -flat, jnp.inf)
    threshold_neg = jnp.sort(neg_score)[jnp.maximum(n_prune_neg - 1, 0)]
    prune_neg = (flat_active & (flat < 0) & (-flat <= threshold_neg)
                 & (n_prune_neg > 0))

    return (prune_pos | prune_neg).reshape(weights.shape)


def smallest_score_prune_mask(score, active, zeta):
    """Prune the ``zeta`` fraction of active positions with the lowest score.

    Sign-agnostic; used for contribution-utility pruning, where the score is
    already non-negative.
    """
    flat = score.reshape(-1)
    flat_active = active.reshape(-1)

    n_prune = (zeta * jnp.sum(flat_active).astype(jnp.float32)).astype(jnp.int32)
    masked = jnp.where(flat_active, flat, jnp.inf)
    threshold = jnp.sort(masked)[jnp.maximum(n_prune - 1, 0)]
    return (flat_active & (flat <= threshold) & (n_prune > 0)).reshape(score.shape)


def random_inactive_mask(active, n_select, key):
    """Select exactly ``n_select`` inactive positions uniformly at random.

    Layer-global: a replacement may land anywhere in the layer, not only in the
    row a connection was removed from. That is what both the SET and DEEP-R
    papers specify, and it is what lets the degree distribution drift away from
    its binomial initialization.

    Selection is by rank rather than by a value threshold, so it returns
    *exactly* ``n_select`` positions. A threshold would be cheaper but ties
    over-select, and float32 uniforms collide readily over a matrix this size
    -- an upward bias of a few connections per event, which compounds over the
    thousands of events in a run.
    """
    score = jnp.where(active, -1.0, jax.random.uniform(key, active.shape))
    flat = score.reshape(-1)
    order = jnp.argsort(-flat)                       # highest score first
    rank = jnp.zeros_like(order).at[order].set(jnp.arange(order.size))
    return (rank.reshape(active.shape) < n_select) & ~active


def bernoulli_inactive_mask(active, n_select, key):
    """Select inactive positions independently, ``n_select`` of them on average.

    The uniform-over-dormant draw that DEEP-R's reactivation calls for, without
    the sort ``random_inactive_mask`` pays to pin the count exactly. The count
    is binomial around ``n_select`` -- about +/-140 on a 20,000-connection draw
    -- which is tolerable here only because the caller derives ``n_select``
    from the *absolute* target count each event. An overshoot is therefore
    absorbed by the next event's deficit rather than accumulating, which is not
    true of a selection rule that is biased in one direction.

    ``n_select`` may be negative after an overshoot; the clip then regrows
    nothing and the excess decays as connections continue to be pruned.
    """
    n_dormant = jnp.maximum((~active).sum(), 1)
    p = jnp.clip(n_select / n_dormant.astype(jnp.float32), 0.0, 1.0)
    return (~active) & jax.random.bernoulli(key, p, active.shape)


def bias_corrected(trace, step, decay):
    """Adam-style bias correction with a floor, for freshly-reset traces."""
    step_f = jnp.maximum(step.astype(jnp.float32), 1.0)
    return trace / jnp.maximum(1.0 - jnp.power(decay, step_f), 1e-12)
