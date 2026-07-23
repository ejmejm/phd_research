import logging
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array 
from optax._src import base

from phd.jax_core.utils import tree_unzip


logger = logging.getLogger(__name__)


class IDBDState(NamedTuple):
    """State for the IDBD algorithm."""
    init_beta: base.Updates
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
    version: str = 'prediction_grads', # {prediction_grads, loss_grads}
    shadow_weight_threshold_factor: float = 0.0,
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
        step_size_decay: Step size decay factor (default: 0.0)
        autostep: Whether to use autostep (default: False)
        tau: Tau parameter for autostep (default: 1e4)
        version: Version of IDBD to use (default: prediction_grads)
        shadow_weight_threshold_factor: When a step-size is less than this value times the initial step-size
            of a given weight, then the step-size will be treated as zero. Only applicable when optimizer is idbd.
    
    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        assert autostep or step_size_decay == 0.0, "Step size decay is only supported with autostep!"
        assert version in ['prediction_grads', 'loss_grads'], f"Invalid IDBD version: {version}!"
        
        if step_size_decay != 0.0:
            raise NotImplementedError("Step-size decay is not implemented with JAX!")
        
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
        
        return IDBDState(init_beta=init_beta, beta=beta, h=h, v=v)

    def update_fn(updates, state, params):
        loss_grads, prediction_grads = updates
        init_beta, beta, h, v = state
        
        # Try square of loss grads version of this as fisher approximation of hessian
        if version == 'loss_grads':
            h_decay_term = jax.tree.map(jnp.square, loss_grads)
        elif version == 'prediction_grads':
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
                
                # If we are using shadow weights, apply the thresholding before computing the effective step-size
                if shadow_weight_threshold_factor > 0.0:
                    thresholded_alpha = jax.tree.map(
                        lambda a_i: jnp.maximum(0.0, a_i - shadow_weight_threshold_factor * jnp.exp(init_beta)),
                        alpha,
                    )
                    raw_effective_step_size = jnp.sum(thresholded_alpha * h_decay_term, axis=-1, keepdims=True)
                else:
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
        
        # Threshold step-sizes for shadow weights
        if shadow_weight_threshold_factor > 0.0:
            alpha = jax.tree.map(
                lambda a_i: jnp.maximum(0.0, a_i - shadow_weight_threshold_factor * jnp.exp(init_beta)),
                alpha,
            )
        
        # Compute parameter updates
        weight_decay_term = jax.tree.map(
            lambda p_i: weight_decay * p_i, params)
        param_updates = jax.tree.map(
            lambda a_i, g_i, w_i: -a_i * (g_i + w_i),
            alpha, loss_grads, weight_decay_term,
        )
        
        # Update state
        state = IDBDState(init_beta=init_beta, beta=beta, h=h, v=v)
        
        return param_updates, state

    return base.GradientTransformation(init_fn, update_fn)


def optax_idbd3(
    meta_lr: float = 0.005,
    init_lr: float = 0.01,
    version: str = 'prediction_grads', # {prediction_grads, loss_grads}
) -> base.GradientTransformation:
    """Incremental Delta-Bar-Delta 3 optimizer.
    
    This is an implementation of the IDBD3 algorithm adapted for deep neural networks.
    Instead of working with input features directly, it uses gradients with respect
    to parameters and maintains separate learning rates for each parameter.
    
    Args:
        params: Iterable of parameters to optimize
        meta_lr: Meta learning rate (default: 0.01)
        init_lr: Initial learning rate (default: 0.01)
        version: Version of IDBD to use (default: prediction_grads)
    
    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        assert version in ['prediction_grads', 'loss_grads'], f"Invalid IDBD version: {version}!"
        
        init_beta = jnp.log(init_lr)
        beta = jax.tree.map(lambda x: jnp.full_like(x, init_beta), params)
        h = jax.tree.map(lambda x: jnp.zeros_like(x), params)

        return IDBDState(init_beta=init_beta, beta=beta, h=h, v=None)

    def update_fn(updates, state, params):
        loss_grads, prediction_grads = updates
        init_beta, beta, h, _ = state
        
        # Try square of loss grads version of this as fisher approximation of hessian
        if version == 'loss_grads':
            h_decay_term = jax.tree.map(jnp.square, loss_grads)
        elif version == 'prediction_grads':
            h_decay_term = jax.tree.map(jnp.square, prediction_grads)
        
        beta = jax.tree.map(
            lambda b_i, g_i, h_i: b_i + meta_lr * g_i * h_i,
            beta, prediction_grads, h,
        )
        alpha = jax.tree.map(jnp.exp, beta)
        
        # Update gradient trace (h)
        h = jax.tree.map(
            lambda h_i, a_i, g_i, d_i: h_i * jnp.clip(1 - a_i * d_i, min=0) + g_i,
            h, alpha, loss_grads, h_decay_term,
        )
        
        # Compute parameter updates
        param_updates = jax.tree.map(
            lambda a_i, g_i: -a_i * g_i,
            alpha, loss_grads,
        )
        
        # Update state
        state = IDBDState(init_beta=init_beta, beta=beta, h=h, v=None)
        
        return param_updates, state

    return base.GradientTransformation(init_fn, update_fn)


def categorical_optax_idbd(
    meta_lr: float = 0.005,
    init_lr: float = 0.01,
    weight_decay: float = 0.0,
    autostep: bool = False,
    tau: float = 1e4,
) -> base.GradientTransformation:
    """
    IDBD for softmax + categorical cross-entropy with linear logits.
    
    Expects updates = (loss_grads, logit_grads, probs)
    where:
        loss_grads  = ∂ℓ/∂w
        logit_grads = ∂z/∂w
        probs       = softmax(z)
    """

    def init_fn(params):
        init_beta = jnp.log(init_lr)
        beta = jax.tree.map(lambda x: jnp.full_like(x, init_beta), params)
        h = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params) if autostep else None
        return IDBDState(init_beta=init_beta, beta=beta, h=h, v=v)

    def update_fn(updates, state, params):
        loss_grads, logit_grads, probs = updates
        beta, h, v = state.beta, state.h, state.v

        alpha = jax.tree.map(jnp.exp, beta)

        # --- curvature term: exact softmax GN diagonal ---
        h_decay_term = jax.tree.map(
            lambda dz, p: jnp.square(dz) * p * (1.0 - p),
            logit_grads,
            probs,
        )

        # --- meta update ---
        if autostep:
            def _autostep(beta, h, v, g, d):
                alpha = jnp.exp(beta)
                v = jnp.maximum(
                    jnp.abs(h * g),
                    v + (alpha / tau) * d * (jnp.abs(h * g) - v),
                )
                new_alpha = alpha * jnp.exp(meta_lr * h * g / v)
                alpha = jnp.where(v > 0, new_alpha, alpha)
                beta = jnp.log(alpha)
                return beta, v

            beta, v = jax.tree.map(_autostep, beta, h, v, loss_grads, h_decay_term)
            alpha = jax.tree.map(jnp.exp, beta)

        else:
            beta = jax.tree.map(
                lambda b, g, h: b + meta_lr * g * h,
                beta, loss_grads, h,
            )
            alpha = jax.tree.map(jnp.exp, beta)

        # --- h-trace update (IDBD core equation) ---
        h = jax.tree.map(
            lambda h, a, g, d: h * jnp.clip(1 - a * d, 0.0) + a * g,
            h, alpha, loss_grads, h_decay_term,
        )

        # --- parameter update ---
        param_updates = jax.tree.map(
            lambda a, g, p: -a * (g + weight_decay * p),
            alpha, loss_grads, params,
        )

        return param_updates, IDBDState(
            init_beta=state.init_beta, beta=beta, h=h, v=v
        )

    return base.GradientTransformation(init_fn, update_fn)
