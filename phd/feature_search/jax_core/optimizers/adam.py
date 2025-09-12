import logging
from typing import NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array 
from optax._src import base

from phd.feature_search.jax_core.utils import tree_unzip


logger = logging.getLogger(__name__)


class AdamState(NamedTuple):
    """State for the IDBD algorithm."""
    lr: base.Updates
    step: base.Updates
    exp_avg: base.Updates
    exp_avg_sq: Optional[base.Updates] = None


def custom_optax_adam(
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> base.GradientTransformation:
    """Adam optimizer with per-step step-sizes.
    
    This is an implementation of the Adam optimizer with per-step step-sizes.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
            
    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        lr = jnp.array(lr, dtype=jnp.float32)
        step = jax.tree.map(lambda x: jnp.zeros(x.shape[-1], dtype=jnp.int32), params)
        exp_avg = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
        exp_avg_sq = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
        
        return AdamState(lr=lr, step=step, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

    def update_fn(updates, state, params):
        loss_grads, prediction_grads = updates
        beta, h, v = state
        
        h_decay_term = jax.tree.map(jnp.square, prediction_grads)
        
        if autostep:
            def _autostep_update(beta, h, v, loss_grads, h_decay_term):
                alpha = jnp.exp(beta)
                v = jnp.maximum(
                    jnp.abs(h * loss_grads),
                    v + 1.0 / tau * alpha * h_decay_term * (jnp.abs(h * loss_grads) - v),
                )
                new_alpha = alpha * jnp.exp(meta_lr * h * loss_grads / v)
                alpha = jnp.where(v != 0, new_alpha, alpha)
                
                raw_effective_step_size = jnp.sum(alpha * h_decay_term, axis=-1, keepdims=True)
                effective_step_size = jnp.clip(raw_effective_step_size, min=1.0)
                
                alpha = alpha / effective_step_size
                beta = jnp.log(alpha)
                
                return alpha, beta, v, raw_effective_step_size.squeeze(-1)
            
            results = jax.tree.map(
                _autostep_update,
                beta, h, v, loss_grads, h_decay_term,
            )
            alpha, beta, v, raw_effective_step_size = tree_unzip(results, 4)
        
        else:
            beta = jax.tree.map(
                lambda b_i, g_i, h_i: b_i + meta_lr * g_i * h_i,
                beta, prediction_grads, h,
            )
            alpha = jax.tree.map(jnp.exp, beta)
        
        # Update gradient trace (h)
        h = jax.tree.map(
            lambda h_i, a_i, g_i, d_i: h_i * jnp.clip(1 - a_i * d_i, min=0) + a_i * g_i,
            h, alpha, loss_grads, h_decay_term,
        )
        
        # Compute parameter updates
        weight_decay_term = jax.tree.map(
            lambda p_i: weight_decay * p_i, params)
        param_updates = jax.tree.map(
            lambda a_i, g_i, w_i: -a_i * (g_i + w_i),
            alpha, loss_grads, weight_decay_term,
        )
        
        # Update state
        state = IDBDState(beta=beta, h=h, v=v)
        
        return param_updates, state

    return base.GradientTransformation(init_fn, update_fn)