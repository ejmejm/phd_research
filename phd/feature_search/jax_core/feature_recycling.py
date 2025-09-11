import random
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from .optimizer import EqxOptimizer


class FeatureStats(eqx.Module):
    age: Int[Array, 'n_features']
    utility: Float[Array, 'n_features']


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
    all_feature_stats: PyTree # Pytree with FeatureStats for leaves
    rng: PRNGKeyArray
    
    def __init__(
        self,
        model: eqx.Module,
        replace_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = 100,
        incoming_weight_init: str = 'kaiming_uniform', # {'kaiming_uniform', 'binary'}
        outgoing_weight_init: str = 'zeros', # {'zeros', 'kaiming_uniform'}
        utility_reset_mode: str = 'median', # {'median', 'zero'}
        initial_step_size_method: str = 'constant', # {'constant', 'mean', 'median'}
        filter_spec: Optional[PyTree] = None,
        rng: Optional[PRNGKeyArray] = None,
    ):
        assert utility_reset_mode in {'median', 'zero'}
        assert incoming_weight_init in {'kaiming_uniform', 'binary'}
        assert outgoing_weight_init in {'zeros', 'kaiming_uniform'}
        assert initial_step_size_method in {'constant', 'mean', 'median'}
        
        if filter_spec is not None:
            model = eqx.filter(model, filter_spec)
        
        is_linear_weights = lambda x: isinstance(x, Array) and x.ndim == 2 
        assert jnp.all(jnp.array(jax.tree.leaves(jax.tree.map(lambda x: is_linear_weights(x), model)))), \
            "All layers must be 2D weight matrices"
        
        self.all_feature_stats = jax.tree.map(
            lambda weights: FeatureStats(
                age = jnp.zeros(weights.shape[1], dtype=jnp.int32),
                utility = jnp.zeros(weights.shape[1], dtype=jnp.float32)
            ),
            tree = model,
        )
        
        self.incoming_weight_init = incoming_weight_init
        self.outgoing_weight_init = outgoing_weight_init
        self.utility_reset_mode = utility_reset_mode
        self.initial_step_size_method = initial_step_size_method
        
        self.replace_rate = jnp.array(replace_rate, dtype=jnp.float32)
        self.decay_rate = jnp.array(decay_rate, dtype=jnp.float32)
        self.maturity_threshold = jnp.array(maturity_threshold, dtype=jnp.int32)
        
        if rng is None:
            rng = jax.random.PRNGKey(random.randint(0, 2**31))
        self.rng = rng
    
    def prune_layer_features(
        self,
        weights: Array,
        path: Tuple[...], # Of types GetArrKey (jax.tree_util.GetAttrKey)
        optimizer: Optional[EqxOptimizer] = None,
    ) -> Array:
        assert weights.ndim == 2, "Weights must be 2D"
        n_features = weights.shape[1]
        
        # Update age
        # Update utility
        # Update replacement accumulator
        
        # Get indices to reinitialize (prune mask)
        prune_mask = jnp.zeros(layer.in_features, dtype=jnp.bool_)
        
        # Reset stats for those features
        # Reinit input and output weights for given features
        
        # Reinit optimizer input and output weight states for given features
        
        return prune_mask
    
    def prune_features(
        self,
        model: eqx.Module,
        activation_values: eqx.Module,
        optimizer: Optional[EqxOptimizer] = None,
    ) -> eqx.Module:
        """Prune features based on CBP utility and return a mask over the features reset.
        
        Args:
            model: The full model to prune
            activation_values: Pytree matching the structure of model with the activation values for each layer
            optimizer: The optimizer optimizing the given model
            filter_spec: Boolean Pytree matching the structure of model with True for prunable layers
            
        Returns:
            A mask over the features reset
        """
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