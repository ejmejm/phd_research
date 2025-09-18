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
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        learning_rate = jnp.array(lr, dtype=jnp.float32)
        step = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.int32), params)
        exp_avg = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
        exp_avg_sq = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
        
        return AdamState(lr=learning_rate, step=step, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

    def update_fn(updates, state, params):
        loss_grads = updates
        lr, step, exp_avg, exp_avg_sq = state
        
        def _adam_update(step, exp_avg, exp_avg_sq, grad):
            step += 1
            
            # Decay the first and second moment running average coefficient 
            exp_avg = exp_avg * betas[0] + grad * (1 - betas[0])
            exp_avg_sq = exp_avg_sq * betas[1] + grad**2 * (1 - betas[1])
            
            # Bias correction
            bias_correction1 = 1 - betas[0]**step
            bias_correction2 = 1 - betas[1]**step

            step_size = lr / bias_correction1
            denom = jnp.sqrt(bias_correction2 / exp_avg_sq) + eps
            param_update = exp_avg / denom * -step_size
            
            return param_update, step, exp_avg, exp_avg_sq
        
        results = jax.tree.map(
            _adam_update,
            step, exp_avg, exp_avg_sq, loss_grads,
        )
        param_updates, step, exp_avg, exp_avg_sq = tree_unzip(results, 4)
        
        # Update state
        state = AdamState(
            lr = lr,
            step = step,
            exp_avg = exp_avg,
            exp_avg_sq = exp_avg_sq,
        )
        
        return param_updates, state

    return base.GradientTransformation(init_fn, update_fn)