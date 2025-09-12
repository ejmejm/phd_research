import random
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from .optimizer import EqxOptimizer
from .utils import tree_replace


class FeatureStats(eqx.Module):
    age: Int[Array, 'n_features']
    utility: Float[Array, 'n_features']
    replacement_accumulator: Float[Array, '']


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
                utility = jnp.zeros(weights.shape[1], dtype=jnp.float32),
                replacement_accumulator = jnp.zeros(1, dtype=jnp.float32),
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
        
    def _compute_new_feature_stats(
        self,
        feature_stats: FeatureStats,
        weights: Float[Array, 'out_features in_features'],
        input_values: Float[Array, 'batch_size in_features'],
    ) -> FeatureStats:
        """Update the feature stats for a single given layer."""
        # Age
        age = feature_stats.age + 1
        
        # Replacement accumulator
        n_features = weights.shape[1]
        replacement_accumulator = feature_stats.replacement_accumulator + self.replace_rate * n_features
        
        # Utility
        weight_sums = jnp.sum(jnp.abs(weights), axis=0) # Shape: (in_features,)
        input_magnitudes = jnp.abs(input_values).mean(axis=0) # Shape: (in_features,)
        step_utility = input_magnitudes * weight_sums
        utility = (1 - self.decay_rate) * step_utility + self.decay_rate * feature_stats.utility
        
        return FeatureStats(
            age = age,
            utility = utility,
            replacement_accumulator = replacement_accumulator,
        )

    def _make_prune_mask(
        self, feature_stats: FeatureStats
    ) -> Tuple[Bool[Array, 'n_features'], Int[Array, '']]:
        """Returns a boolean mask of which features to prune and the number of features to prune."""
        
        # Determine which features are eligible for replacement, and which to replace
        eligibility_mask = feature_stats.age > self.maturity_threshold
        n_eligible_replacements = jnp.sum(eligibility_mask)
        n_available_replacements = feature_stats.replacement_accumulator.astype(jnp.int32)
        n_replacements = jnp.minimum(n_available_replacements, n_eligible_replacements)
        
        # Compute the threshold for pruning
        filtered_utility = jnp.where(eligibility_mask, feature_stats.utility, jnp.inf)
        utility_ranking = jnp.argsort(filtered_utility)
        utility_threshold = filtered_utility[utility_ranking[n_replacements - 1]]
        
        # Construct the prune mask
        prune_mask = jnp.where(filtered_utility <= utility_threshold, True, False)
        prune_mask = prune_mask & eligibility_mask
        
        return prune_mask, n_replacements
    
    def prune_layer_features(
        self,
        weights: Float[Array, 'out_features in_features'],
        input_values: Float[Array, 'batch_size in_features'],
        feature_stats: FeatureStats,
        # path: Tuple[...], # Of types GetArrKey (jax.tree_util.GetAttrKey)
        optimizer_state: PyTree = None,
    ) -> Tuple[FeatureStats, Optional[EqxOptimizer], Array]:
        assert weights.ndim == 2, "Weights must be 2D"
        n_features = weights.shape[1]
        
        feature_stats = self._compute_new_feature_stats(feature_stats, weights, input_values)
        
        # Get indices to reinitialize (prune mask)
        prune_mask, n_replacements = self._make_prune_mask(feature_stats)
        feature_stats = tree_replace(
            feature_stats,
            replacement_accumulator = feature_stats.replacement_accumulator - n_replacements,
        )
        
        # Reset stats for those features
        # Reinit input and output weights for given features
        
        # Reinit optimizer input and output weight states for given features
        
        return feature_stats, optimizer_state, prune_mask
    
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
        # Tree map prune_layer_features to each set of weights in the model
        pass
        
        
        # reset_idxs = {}
        # for layer in self._tracked_layers.keys():
        #     self._step_replacement_accumulator(layer)
        #     layer_reset_idxs = self._get_layer_prune_idxs(layer)
        #     self._prune_layer(layer, layer_reset_idxs)
        #     if layer_reset_idxs is not None and len(layer_reset_idxs) > 0:
        #         reset_idxs[layer] = layer_reset_idxs
        # return reset_idxs
    
    def track(self, layer: eqx.Module):
        pass
    
    def get_statistics(self, layer: eqx.Module):
        pass