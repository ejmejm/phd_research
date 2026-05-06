import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
from optax._src import base


logger = logging.getLogger(__name__)


class LionLionMetaOptimizeState(NamedTuple):
    """State for the Lion+Lion MetaOptimize algorithm with weight-level step-sizes."""
    init_beta: base.Updates
    beta: base.Updates       # log step-size, one per weight
    m: base.Updates          # base Lion momentum
    meta_m: base.Updates     # meta Lion momentum (per weight)
    h: base.Updates          # discounted trace of past parameter updates


# NOTE: This has not yet been extensively tested, and the optimizer may have problems
def optax_lion_lion_metaoptimize(
    meta_lr: float = 1e-3,
    init_lr: float = 1e-3,
    gamma: float = 0.9999,
    rho: float = 0.9,
    meta_rho: float = 0.99,
    c: float = 0.9,
    meta_c: float = 0.9,
    weight_decay: float = 0.0,
) -> base.GradientTransformation:
    """Lion + Lion MetaOptimize (Algorithm 3, Sharifnassab et al. 2024) with
    weight-level step-sizes.

    Both the base optimizer and the meta optimizer are Lion. Each weight has
    its own log step-size beta, updated by a per-weight Lion meta-update on
    the meta-gradient z = h * grad, where h is a discounted trace of past
    parameter updates.

    Args:
        meta_lr: Meta step-size eta for the Lion meta-update on beta.
        init_lr: Initial per-weight step-size alpha_0; beta is initialised
            to log(init_lr).
        gamma: Discount factor for the h trace (<= 1).
        rho: Decay rate for the base Lion momentum.
        meta_rho: Decay rate for the meta Lion momentum.
        c: Base Lion sign-mixing coefficient (interpolates between momentum
            and current gradient inside sign()).
        meta_c: Meta Lion sign-mixing coefficient.
        weight_decay: Decoupled weight decay coefficient kappa, scaled by
            alpha (Lion-W style).

    Returns:
        An :class:`optax.GradientTransformation` whose updates expect plain
        loss gradients.
    """

    def init_fn(params):
        init_beta = jnp.log(jnp.asarray(init_lr, dtype=jnp.float32))
        beta = jax.tree.map(lambda x: jnp.full_like(x, init_beta), params)
        m = jax.tree.map(jnp.zeros_like, params)
        meta_m = jax.tree.map(jnp.zeros_like, params)
        h = jax.tree.map(jnp.zeros_like, params)
        return LionLionMetaOptimizeState(
            init_beta=init_beta, beta=beta, m=m, meta_m=meta_m, h=h,
        )

    def update_fn(grads, state, params):
        init_beta, beta, m, meta_m, h = state

        # --- Meta Lion update on beta ------------------------------------
        # Per-weight meta-gradient: z = h_t * g_t (elementwise).
        z = jax.tree.map(lambda h_i, g_i: h_i * g_i, h, grads)

        new_meta_m = jax.tree.map(
            lambda mm_i, z_i: meta_rho * mm_i + (1.0 - meta_rho) * z_i,
            meta_m, z,
        )
        meta_direction = jax.tree.map(
            lambda mm_i, z_i: jnp.sign(meta_c * mm_i + (1.0 - meta_c) * z_i),
            new_meta_m, z,
        )
        new_beta = jax.tree.map(
            lambda b_i, d_i: b_i - meta_lr * d_i,
            beta, meta_direction,
        )
        alpha = jax.tree.map(jnp.exp, new_beta)

        # --- Base Lion update on parameters ------------------------------
        new_m = jax.tree.map(
            lambda m_i, g_i: rho * m_i + (1.0 - rho) * g_i,
            m, grads,
        )
        base_direction = jax.tree.map(
            lambda m_i, g_i: jnp.sign(c * m_i + (1.0 - c) * g_i),
            new_m, grads,
        )
        param_updates = jax.tree.map(
            lambda a_i, d_i, p_i: -a_i * d_i - weight_decay * a_i * p_i,
            alpha, base_direction, params,
        )

        # --- Update h trace with the actual parameter update -------------
        new_h = jax.tree.map(
            lambda h_i, a_i, dw_i: gamma * (1.0 - weight_decay * a_i) * h_i + dw_i,
            h, alpha, param_updates,
        )

        new_state = LionLionMetaOptimizeState(
            init_beta=init_beta, beta=new_beta, m=new_m, meta_m=new_meta_m, h=new_h,
        )
        return param_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)
