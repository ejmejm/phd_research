import random
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from .optimizer import EqxOptimizer


class FeatureStats(eqx.Module):
    age: Int[Array, '']
    utility: Float[Array, '']


class CBPTracker(eqx.Module):
    """Class that performs continual backprop"""
    
    # Static
    incoming_weight_init: str = eqx.field(static=True)
    outgoing_weight_init: str = eqx.field(static=True)
    utility_reset_mode: str = eqx.field(static=True)
    initial_step_size_method: str = eqx.field(static=True)
    
    # Non-static
    replace_rate: Float[Array, '']
    decay_rate: Float[Array, '']
    maturity_threshold: Int[Array, '']
    rng: PRNGKeyArray
    
    def __init__(
        self,
        replace_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = 100,
        incoming_weight_init: str = 'kaiming_uniform', # {'kaiming_uniform', 'binary'}
        outgoing_weight_init: str = 'zeros', # {'zeros', 'kaiming_uniform'}
        utility_reset_mode: str = 'median', # {'median', 'zero'}
        initial_step_size_method: str = 'constant', # {'constant', 'mean', 'median'}
        rng: Optional[PRNGKeyArray] = None,
    ):
        assert utility_reset_mode in {'median', 'zero'}
        assert incoming_weight_init in {'kaiming_uniform', 'binary'}
        assert outgoing_weight_init in {'zeros', 'kaiming_uniform'}
        assert initial_step_size_method in {'constant', 'mean', 'median'}
        
        if rng is None:
            rng = jax.random.PRNGKey(random.randint(0, 2**31))
        
        self.incoming_weight_init = incoming_weight_init
        self.outgoing_weight_init = outgoing_weight_init
        self.utility_reset_mode = utility_reset_mode
        self.initial_step_size_method = initial_step_size_method
        
        self.replace_rate = jnp.array(replace_rate, dtype=jnp.float32)
        self.decay_rate = jnp.array(decay_rate, dtype=jnp.float32)
        self.maturity_threshold = jnp.array(maturity_threshold, dtype=jnp.int32)
        self.rng = rng
    
    def prune_layer_features(
        self,
        layer: eqx.Module,
        reset_idxs: List[int],
    ) -> Array:
        pass
    
    def prune_features(
        self,
        model: eqx.Module,
        optimizer: Optional[EqxOptimizer] = None,
        filter_spec: Optional[PyTree] = None,
    ) -> eqx.Module:
        """Prune features based on CBP utility and return a mask over the features reset."""
        reset_idxs = {}
        for layer in self._tracked_layers.keys():
            self._step_replacement_accumulator(layer)
            layer_reset_idxs = self._get_layer_prune_idxs(layer)
            self._prune_layer(layer, layer_reset_idxs)
            if layer_reset_idxs is not None and len(layer_reset_idxs) > 0:
                reset_idxs[layer] = layer_reset_idxs
        return reset_idxs
    
    def track(self, layer: eqx.Module):
        pass
    
    def get_statistics(self, layer: eqx.Module):
        pass