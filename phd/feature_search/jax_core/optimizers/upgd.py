import logging
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from optax._src import base

from phd.feature_search.jax_core.utils import tree_unzip

logger = logging.getLogger(__name__)


class UPGDState(NamedTuple):
    """State for the UPGD algorithm."""
    step: jnp.ndarray
    avg_utility: base.Updates
    rng_key: Optional[PRNGKeyArray] = None


def optax_upgd(
    lr: float = 1e-5,
    weight_decay: float = 0.001,
    beta_utility: float = 0.999,
    sigma: float = 0.001,
) -> base.GradientTransformation:
    """Utility-based Parameter Gradient Descent optimizer.
    
    This optimizer uses utility-based scaling to adaptively adjust learning rates
    for each parameter based on their utility (negative gradient-parameter product).
    
    Args:
        lr: Learning rate (default: 1e-5)
        weight_decay: Weight decay coefficient (default: 0.001)
        beta_utility: Exponential decay factor for utility averaging (default: 0.999)
        sigma: Standard deviation for noise injection (default: 0.001)
            
    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        step = jnp.array(0, dtype=jnp.int32)
        avg_utility = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        rng_key = jax.random.key(0)
        return UPGDState(step=step, avg_utility=avg_utility, rng_key=rng_key)

    def update_fn(updates, state, params):
        grads = updates
        step, avg_utility, rng_key = state
        
        # Update step counter
        step = step + 1
        
        # First pass: update avg_utility and compute max utility per parameter
        def _update_utility_and_get_max(avg_util, grad, param):
            # Update average utility: avg_utility = beta * avg_utility + (1 - beta) * (-grad * param)
            new_avg_utility = avg_util * beta_utility + (1 - beta_utility) * (-grad * param)
            # Get max utility for this parameter
            max_util = jnp.max(new_avg_utility)
            return new_avg_utility, max_util
        
        # Update avg_utility and collect max utilities
        # jax.tree.map returns a tree of tuples, so we need to unzip it
        results = jax.tree.map(
            _update_utility_and_get_max,
            avg_utility, grads, params,
        )
        updated_avg_utility, max_utils = tree_unzip(results, 2)
        
        # Find global maximum utility across all parameters
        global_max_util = jax.tree.reduce(
            lambda x, y: jnp.maximum(x, y),
            max_utils,
            initializer=jnp.array(-jnp.inf),
        )
        
        # Split PRNG key for each parameter leaf
        num_leaves = len(jax.tree.leaves(params))
        rng_keys = jax.random.split(rng_key, num_leaves)
        # Map keys to tree structure using unflatten
        tree_def = jax.tree.structure(params)
        rng_key_tree = jax.tree.unflatten(tree_def, rng_keys)
        
        # Second pass: update parameters using scaled utility
        def _update_params(avg_util, grad, param, rng_key):
            # Bias correction for utility (step is a scalar, so this broadcasts correctly)
            bias_correction_utility = 1 - beta_utility ** step
            
            # Generate noise
            noise = jax.random.normal(rng_key, shape=grad.shape, dtype=grad.dtype) * sigma
            
            # Compute scaled utility: sigmoid((avg_utility / bias_correction) / global_max_util)
            normalized_utility = (avg_util / bias_correction_utility) / global_max_util
            scaled_utility = jax.nn.sigmoid(normalized_utility)
            
            # Update parameter: param = param * (1 - lr * weight_decay) - 2.0 * lr * (grad + noise) * (1 - scaled_utility)
            param_update = param * (1 - lr * weight_decay) - 2.0 * lr * (grad + noise) * (1 - scaled_utility)
            
            return param_update
        
        # Update parameters
        param_updates = jax.tree.map(
            _update_params,
            updated_avg_utility, grads, params, rng_key_tree,
        )
        
        # Compute actual updates (difference from current params)
        updates = jax.tree.map(
            lambda new_p, old_p: new_p - old_p,
            param_updates, params,
        )
        
        # Update PRNG key for next iteration
        _, new_rng_key = jax.random.split(rng_key)
        
        # Update state
        state = UPGDState(step=step, avg_utility=updated_avg_utility, rng_key=new_rng_key)
        
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
