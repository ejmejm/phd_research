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
land anywhere in the layer. The **per-layer connection count is what is
preserved** -- note that Algorithm 1 states a single global ``K`` and it is the
authors' released code that enforces the budget per weight matrix; this follows
the released code. ``algorithms/dynamic/deep_r.py``
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
    ``deficit / n_dormant``. The paper restores the count exactly (Algorithm 1
    line 6, "while number of active connections lower than K"); this lands
    near it instead, with a per-event standard deviation of about the square
    root of the number of deaths in that event -- a handful of connections out
    of 203,264 at ``event_period: 1``. It self-corrects, because the deficit is
    measured against the absolute target every event. Pinning it
    exactly would mean sorting the whole matrix, which costs 34x more per
    event and is what forces ``event_period`` above 1.

The per-connection *update rule* is therefore unchanged at any
``event_period``. The dynamics as a whole are not: a connection left dormant
for longer is absent from the forward pass for longer, which changes every
gradient. Raising it is a cost/accuracy trade, and the paper's own drifting-task
experiment argues against it -- Appendix A: "we enhanced the noise exploration
by setting a batch to 1 so that the connectivity matrices were updated at every
time step". ``event_period: 1`` is the default here and what the sweeps use.

One deviation remains: the L1 and noise terms are applied after the optimizer
step rather than folded into it. For SGD these are identical; for a
preconditioned optimizer they are not, so ``optimizer.name: sgd`` is the
supported setting.
"""

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
    """Per-layer connection counts, and the fixed sign of every position.

    The counts are measured once at initialization; taking the target from the
    initial topology rather than the current one lets pruning and regrowth
    happen at different times without the budget drifting.

    The signs implement the paper's parameterization ``w = s * theta`` with
    ``theta >= 0``: every position in the matrix -- active or dormant -- is
    assigned a sign once and keeps it for the whole run, so a position can only
    ever hold a connection of that sign. That is what makes DEEP-R a sampler
    over a fixed hypothesis space rather than a free search. Stored as int8:
    at H=1024 a float32 copy would be 51 MB per seed.
    """
    target_w1: jax.Array   # scalar
    target_w2: jax.Array   # scalar
    sign_w1: jax.Array     # (max_hidden, input_dim) int8, +-1
    sign_w2: jax.Array     # (output_dim, max_hidden) int8, +-1
    #: Connections that went dormant since the last event. Counted rather than
    #: inferred from the reactivation count, so that the two logged series are
    #: independent measurements and cross-checking turnover means something.
    n_pruned: jax.Array    # scalar


def _init_signs(weights, mask, key):
    """Assign each position its permanent sign.

    Active positions keep the sign their initial weight already carries. That
    is distributionally identical to the paper's ``theta = |w_0|`` with an
    independent random sign, because ``w_0`` is symmetric about zero -- so no
    re-signing of the initial weights is needed. Dormant positions get a fresh
    draw, which is the sign they will carry if they are ever reactivated.
    """
    drawn = jnp.where(jax.random.bernoulli(key, 0.5, weights.shape), 1, -1)
    here = jnp.where(weights >= 0, 1, -1)
    return jnp.where(mask.astype(jnp.bool_), here, drawn).astype(jnp.int8)


def _flag(weights, mask, sign, key, *, lr, l1, temperature):
    """Advance the Langevin dynamics and deactivate connections whose theta < 0.

    Only the L1 term and the death test consult the sign. The gradient step
    does not need to: with ``w = s * theta`` and ``s**2 = 1``, a step on theta
    is exactly a step on w, and the Gaussian noise is sign-symmetric. So the
    optimizer is left alone and this is the whole of the sign constraint.
    """
    active = mask.astype(jnp.bool_)
    s = sign.astype(weights.dtype)

    noise = jax.random.normal(key, weights.shape) * jnp.sqrt(2.0 * lr * temperature)
    noisy = jnp.where(active, weights - lr * l1 * s + noise, weights)

    # theta < 0  <=>  s * w < 0. theta == 0 is the boundary and stays active.
    dead = active & (s * noisy < 0)
    return (jnp.where(dead, 0.0, noisy),
            (active & ~dead).astype(mask.dtype),
            dead.sum().astype(jnp.int32))


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
        event_period: int = 1,
        regrow_theta: float = 1e-12,
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
            regrow_theta: Magnitude a reactivated connection re-enters at. The
                reference implementation uses 1e-12, i.e. effectively zero but
                on the correct side of the sign boundary. It cannot be exactly
                0: with a fixed sign, theta = 0 sits on the boundary and the
                next L1 step would push it straight back to dormant.
        """
        self.learning_rate = float(learning_rate)
        self.l1 = float(l1)
        self.temperature = float(temperature)
        self.evolve_w2 = bool(evolve_w2)
        self.event_period = int(event_period)
        self.regrow_theta = float(regrow_theta)

    def init_state(self, model: PaddedMLP, *, key: PRNGKeyArray) -> DeepRState:
        k1, k2 = jax.random.split(key)
        return DeepRState(
            target_w1=model.w1_mask.sum().astype(jnp.int32),
            target_w2=model.w2_mask.sum().astype(jnp.int32),
            sign_w1=_init_signs(model.W1, model.w1_mask, k1),
            sign_w2=_init_signs(model.W2, model.w2_mask, k2),
            n_pruned=jnp.int32(0),
        )

    def step_update(self, model: PaddedMLP, algo_state, *, key):
        k1, k2 = jax.random.split(key)
        W1, w1_mask, d1 = _flag(
            model.W1, model.w1_mask, algo_state.sign_w1, k1,
            lr=self.learning_rate, l1=self.l1, temperature=self.temperature)

        if self.evolve_w2:
            W2, w2_mask, d2 = _flag(
                model.W2, model.w2_mask, algo_state.sign_w2, k2,
                lr=self.learning_rate, l1=self.l1, temperature=self.temperature)
        else:
            W2, w2_mask, d2 = model.W2, model.w2_mask, jnp.int32(0)

        new_state = tree_replace(algo_state, n_pruned=algo_state.n_pruned + d1 + d2)
        return tree_replace(
            model, W1=W1, W2=W2, w1_mask=w1_mask, w2_mask=w2_mask), new_state

    def event(self, model: PaddedMLP, optimizer, algo_state, *, key):
        k1, k2 = jax.random.split(key)

        # Only slots belonging to an active hidden unit may be reactivated.
        eligible_w1 = model.unit_mask[:, None] > 0
        eligible_w2 = model.unit_mask[None, :] > 0

        active_w1 = model.w1_mask.astype(jnp.bool_)
        active_w2 = model.w2_mask.astype(jnp.bool_)
        regrow_w1 = bernoulli_inactive_mask(
            active_w1 | ~eligible_w1, algo_state.target_w1 - active_w1.sum(), k1)

        if self.evolve_w2:
            regrow_w2 = bernoulli_inactive_mask(
                active_w2 | ~eligible_w2, algo_state.target_w2 - active_w2.sum(), k2)
        else:
            regrow_w2 = jnp.zeros_like(active_w2)

        # Reactivated connections re-enter at theta = regrow_theta, carrying
        # the sign their position was assigned at initialization -- not one
        # chosen by the first gradient. A position whose fixed sign is wrong
        # for the task will simply be pruned again, which is the constraint
        # doing its job.
        new_model = tree_replace(
            model,
            W1=jnp.where(
                regrow_w1, algo_state.sign_w1.astype(model.W1.dtype) * self.regrow_theta,
                model.W1),
            W2=jnp.where(
                regrow_w2, algo_state.sign_w2.astype(model.W2.dtype) * self.regrow_theta,
                model.W2),
            w1_mask=(active_w1 | regrow_w1).astype(model.w1_mask.dtype),
            w2_mask=(active_w2 | regrow_w2).astype(model.w2_mask.dtype),
        )
        # Reset optimizer state on the newly-live slots and on everything
        # dormant, which sweeps up connections flagged since the last event.
        new_optimizer = reset_optimizer_at(
            optimizer, regrow_w1 | ~active_w1, regrow_w2 | ~active_w2)

        n_regrown = regrow_w1.sum().astype(jnp.int32) + regrow_w2.sum().astype(jnp.int32)
        info = {'pruned': algo_state.n_pruned, 'regrown': n_regrown}
        return (new_model, new_optimizer,
                tree_replace(algo_state, n_pruned=jnp.int32(0)), info)
