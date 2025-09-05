import logging
from typing import NamedTuple, Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float 
import optax
from optax._src import base


logger = logging.getLogger(__name__)


class IDBDState(NamedTuple):
    """State for the IDBD algorithm."""
    beta: base.Updates
    h: base.Updates
    v: Optional[base.Updates] = None


def optax_idbd(
    meta_lr: float = 0.005,
    init_lr: float = 0.01,
    weight_decay: float = 0.0,
    step_size_decay: float = 0.0,
    autostep: bool = False,
    tau: float = 1e4,
) -> base.GradientTransformation:
    """Incremental Delta-Bar-Delta optimizer.
    
    This is an implementation of the IDBD algorithm adapted for deep neural networks.
    Instead of working with input features directly, it uses gradients with respect
    to parameters and maintains separate learning rates for each parameter.
    
    Args:
        params: Iterable of parameters to optimize
        meta_lr: Meta learning rate (default: 0.01)
        init_lr: Initial learning rate (default: 0.01)
        weight_decay: Weight decay (default: 0.0)
        version: Version of IDBD to use (default: squared_inputs)
        autostep: Whether to use autostep (default: False)
        tau: Tau parameter for autostep (default: 1e4)\
            
    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        assert autostep or step_size_decay == 0.0, "Step size decay is only supported with autostep!"
        
        if autostep:
            # Check that parameters match a linear layer structure
            weights = eqx.filter(params, lambda x: isinstance(x, Array) and x.ndim in (2, 3))
            biases = eqx.filter(params, lambda x: isinstance(x, Array) and x.ndim == 1)
            
            n_weights = len(jax.tree.leaves(weights))
            n_biases = len(jax.tree.leaves(biases))
            
            assert n_biases == 0, "AutoStep optimizer does not support biases!"
            assert n_weights > 0, "No valid weight parameters found!"

            if n_weights > 1:
                logger.warning(
                    "Found multiple sets of weights, but AutoStep does not support non-linear  "
                    "layer structures. If the weights provided to AutoStep are stacked and not "
                    "independent, then this will probably cause a silent bug."
                )
        
        init_beta = jnp.log(init_lr)
        beta = jax.tree.map(lambda x: jnp.full_like(x, init_beta), params)
        h = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        
        v = None
        if autostep:
            v = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        
        return IDBDState(beta=beta, h=h, v=v)

    def update_fn(updates, state, params):
        loss_grads, prediction_grads = updates
        beta, h, v = state
        
        h_decay_term = jax.tree.map(jnp.square, prediction_grads)
        
        if autostep:
            pass
        
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