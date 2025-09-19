import logging
import random
from typing import Dict, List, NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from .models import lecun_uniform
from .optimizers import EqxOptimizer
from .optimizers.adam import AdamState
from .optimizers.idbd import IDBDState
from .utils import tree_replace, tree_unzip


logger = logging.getLogger(__name__)


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
        incoming_weight_init: str = 'lecun_uniform', # {'lecun_uniform', 'kaiming_uniform', 'binary'}
        outgoing_weight_init: str = 'zeros', # {'zeros', 'lecun_uniform', 'kaiming_uniform'}
        utility_reset_mode: str = 'median', # {'median', 'zero'}
        initial_step_size_method: str = 'constant', # {'constant', 'mean', 'median'}
        filter_spec: Optional[PyTree] = None,
        rng: Optional[PRNGKeyArray] = None,
    ):
        assert utility_reset_mode in {'median', 'zero'}
        assert incoming_weight_init in {'lecun_uniform', 'kaiming_uniform', 'binary'}
        assert outgoing_weight_init in {'zeros', 'kaiming_uniform'}
        assert initial_step_size_method in {'constant', 'mean', 'median'}
        
        if incoming_weight_init == 'kaiming_uniform':
            logger.warning(
                "Kaiming uniform weight initialization is deprecated in the JAX implementation."
                "Using lecun_uniform instead.",
            )
            incoming_weight_init = 'lecun_uniform'
            
        if outgoing_weight_init == 'kaiming_uniform':
            logger.warning(
                "Kaiming uniform weight initialization is deprecated in the JAX implementation."
                "Using lecun_uniform instead.",
            )
            outgoing_weight_init = 'lecun_uniform'
        
        if filter_spec is not None:
            model = eqx.filter(model, filter_spec)
        
        is_linear_weights = lambda x: isinstance(x, Array) and x.ndim == 2 
        assert jnp.all(jnp.array(jax.tree.leaves(jax.tree.map(lambda x: is_linear_weights(x), model)))), \
            "All layers must be 2D weight matrices"
        
        self.all_feature_stats = [
            FeatureStats(
                age = jnp.zeros(weights.shape[1], dtype=jnp.int32),
                utility = jnp.zeros(weights.shape[1], dtype=jnp.float32),
                replacement_accumulator = jnp.array(0.0, dtype=jnp.float32),
            )
            for weights in jax.tree.leaves(model)[1:]
        ]
        
        # jax.tree.map(
        #     lambda weights: FeatureStats(
        #         age = jnp.zeros(weights.shape[1], dtype=jnp.int32),
        #         utility = jnp.zeros(weights.shape[1], dtype=jnp.float32),
        #         replacement_accumulator = jnp.zeros(1, dtype=jnp.float32),
        #     ),
        #     tree = model,
        # )
        
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
        self, feature_stats: FeatureStats, rng: PRNGKeyArray,
    ) -> Tuple[Bool[Array, 'n_features'], Int[Array, '']]:
        """Returns a boolean mask of which features to prune and the number of features to prune."""
        
        # Determine which features are eligible for replacement, and which to replace
        n_available_replacements = feature_stats.replacement_accumulator.astype(jnp.int32)
        
        def _make_mask():
            eligibility_mask = feature_stats.age > self.maturity_threshold
            n_eligible_replacements = jnp.sum(eligibility_mask)
            n_replacements = jnp.minimum(n_available_replacements, n_eligible_replacements)
            
            # Compute the threshold for pruning
            # Perturb the utility to avoid ties
            perturbed_utility = feature_stats.utility + \
                jax.random.normal(rng, feature_stats.utility.shape) * 1e-12
            filtered_utility = jnp.where(eligibility_mask, perturbed_utility, jnp.inf)
            utility_ranking = jnp.argsort(filtered_utility)
            utility_threshold = filtered_utility[utility_ranking[n_replacements - 1]]
            
            # Construct the prune mask
            prune_mask = jnp.where(filtered_utility <= utility_threshold, True, False)
            prune_mask = prune_mask & eligibility_mask
            
            return prune_mask, n_replacements
        
        # TODO: Test how much of a difference this makes in performance
        prune_mask, n_replacements = jax.lax.cond(
            n_available_replacements > 0,
            _make_mask,
            lambda: (jnp.zeros(feature_stats.utility.shape, dtype=jnp.bool_), 0)
        )
        
        return prune_mask, n_replacements
    
    def _reset_feature_stats(self, feature_stats: FeatureStats, prune_mask: Bool[Array, 'n_features']):
        """Resets the feature stats for the given layer and indices."""
        age = jnp.where(prune_mask, 0, feature_stats.age)

        if self.utility_reset_mode == 'median':
            reset_val = jnp.median(feature_stats.utility)
        elif self.utility_reset_mode == 'zero':
            reset_val = 0
        else:
            raise ValueError(f"Invalid utility reset mode: {self.utility_reset_mode}")
        utility = jnp.where(prune_mask, reset_val, feature_stats.utility)
        
        return tree_replace(
            feature_stats,
            age = age,
            utility = utility,
        )
    
    def _reinit_input_weights(
        self,
        in_weights: Float[Array, 'n_features in_features'],
        prune_mask: Bool[Array, 'n_features'],
        rng: PRNGKeyArray,
    ):
        """Selectively reinitialize the weights that output the features of interest."""    
        if self.incoming_weight_init == 'lecun_uniform':
            new_in_weights = lecun_uniform(rng, in_weights.shape)
        elif self.incoming_weight_init == 'binary':
            new_in_weights = jax.random.randint(rng, in_weights.shape, 0, 2).astype(jnp.float32) * 2.0 - 1.0
        else:
            raise ValueError(f"Invalid weight initialization: {self.incoming_weight_init}")
        
        return jnp.where(jnp.expand_dims(prune_mask, 1), new_in_weights, in_weights)
    
    def _reinit_output_weights(
        self,
        out_weights: Float[Array, 'out_features n_features'],
        prune_mask: Bool[Array, 'n_features'],
        rng: PRNGKeyArray,
    ):
        """Selectively reinitialize the weights that output the features of interest."""    
        if self.outgoing_weight_init == 'zeros':
            new_out_weights = jnp.zeros_like(out_weights)
        elif self.outgoing_weight_init == 'lecun_uniform':
            new_out_weights = lecun_uniform(rng, out_weights.shape)
        else:
            raise ValueError(f"Invalid weight initialization: {self.outgoing_weight_init}")

        return jnp.where(jnp.expand_dims(prune_mask, 0), new_out_weights, out_weights)
    
    
    def _reset_input_optim_state(
        self,
        optim_layer_state: Optional[NamedTuple],
        prune_mask: Bool[Array, 'n_features'],
    ) -> Optional[NamedTuple]:
        """Reset the optimizer state for the weights that output features at the given indices."""
        if optim_layer_state is None:
            return None
        
        if isinstance(optim_layer_state, IDBDState):
            mean_beta = jnp.mean(optim_layer_state.beta)
            median_beta = jnp.median(optim_layer_state.beta)
        
        prune_mask = jnp.expand_dims(prune_mask, 1)
        
        new_vals = []
        for i, value in enumerate(optim_layer_state):
            if value.ndim == 2:
                new_vals.append(jnp.where(prune_mask, 0, value))
            else:
                logger.warning(
                    f"Not resetting optimizer state for field `{optim_layer_state._fields[i]}` because ndim != 2 "
                    f"(not linear weights), ndim: {value.ndim}"
                )
                new_vals.append(value)
        
        if isinstance(optim_layer_state, IDBDState):
            beta_idx = optim_layer_state._fields.index('beta')
            if self.initial_step_size_method == 'constant':
                new_vals[beta_idx] = jnp.where(prune_mask, optim_layer_state.init_beta, new_vals[beta_idx])
            elif self.initial_step_size_method == 'mean':
                new_vals[beta_idx] = jnp.where(prune_mask, mean_beta, new_vals[beta_idx])
            elif self.initial_step_size_method == 'median':
                new_vals[beta_idx] = jnp.where(prune_mask, median_beta, new_vals[beta_idx])
            else:
                raise ValueError(f'Invalid initial step-size method: {self.initial_step_size_method}')
        
        return optim_layer_state.__class__(*new_vals)
    
    
    def _reset_output_optim_state(
        self,
        optim_layer_state: Optional[NamedTuple],
        prune_mask: Bool[Array, 'n_features'],
    ) -> Optional[NamedTuple]:
        """Reset the optimizer state for the weights that take in features at the given indices."""
        if optim_layer_state is None:
            return None
        
        
        # Get mean and median beta per output unit
        # Use mean/median per output unit because different units may be moving
        # at different rates.
        if isinstance(optim_layer_state, IDBDState):
            mean_betas = jnp.mean(optim_layer_state.beta, axis=1, keepdims=True)
            median_betas = jnp.median(optim_layer_state.beta, axis=1, keepdims=True)
        
        prune_mask = jnp.expand_dims(prune_mask, 0)
        
        new_vals = []
        for i, value in enumerate(optim_layer_state):
            if value.ndim == 2:
                new_vals.append(jnp.where(prune_mask, 0, value))
            else:
                logger.warning(
                    f"Not resetting optimizer state for field `{optim_layer_state._fields[i]}` because ndim != 2 "
                    f"(not linear weights), ndim: {value.ndim}"
                )
                new_vals.append(value)
        
        if isinstance(optim_layer_state, IDBDState):
            beta_idx = optim_layer_state._fields.index('beta')
            if self.initial_step_size_method == 'constant':
                new_vals[beta_idx] = jnp.where(prune_mask, optim_layer_state.init_beta, new_vals[beta_idx])
            elif self.initial_step_size_method == 'mean':
                new_vals[beta_idx] = jnp.where(prune_mask, mean_betas, new_vals[beta_idx])
            elif self.initial_step_size_method == 'median':
                new_vals[beta_idx] = jnp.where(prune_mask, median_betas, new_vals[beta_idx])
            else:
                raise ValueError(f'Invalid initial step-size method: {self.initial_step_size_method}')
        
        return optim_layer_state.__class__(*new_vals)

    
    def prune_layer_features(
        self,
        in_weights: Float[Array, 'n_features in_features'],
        out_weights: Float[Array, 'out_features n_features'],
        activation_values: Float[Array, 'batch_size n_features'],
        feature_stats: FeatureStats,
        in_optim_state: Optional[NamedTuple] = None,
        out_optim_state: Optional[NamedTuple] = None,
        *,
        rng: PRNGKeyArray,
    ) -> Tuple[FeatureStats, Optional[EqxOptimizer], Array]:
        assert in_weights.ndim == 2, "Weights must be 2D"
        assert out_weights.ndim == 2, "Weights must be 2D"
        n_features = out_weights.shape[1]
        
        in_weight_key, out_weight_key, prune_mask_key = jax.random.split(rng, 3)
        
        # Update feature stats
        feature_stats = self._compute_new_feature_stats(feature_stats, out_weights, activation_values)
        
        # Get indices to reinitialize (prune mask)
        prune_mask, n_replacements = self._make_prune_mask(feature_stats, prune_mask_key)
        feature_stats = tree_replace(
            feature_stats,
            replacement_accumulator = feature_stats.replacement_accumulator - n_replacements,
        )
        
        # TODO: Add optimization that doesn't do this if n_replacements is 0
        
        # Reset stats for those features
        feature_stats = self._reset_feature_stats(feature_stats, prune_mask)
        
        # Reinit input and output weights for given features
        nin_weights = self._reinit_input_weights(in_weights, prune_mask, in_weight_key)
        nout_weights = self._reinit_output_weights(out_weights, prune_mask, out_weight_key)
        
        # Reinit optimizer input and output weight states for given features
        nin_optim_state = self._reset_input_optim_state(in_optim_state, prune_mask)
        nout_optim_state = self._reset_output_optim_state(out_optim_state, prune_mask)
        
        in_weights = nin_weights
        out_weights = nout_weights
        in_optim_state = nin_optim_state
        out_optim_state = nout_optim_state
        
        return feature_stats, in_weights, out_weights, in_optim_state, out_optim_state, prune_mask
    
    # TODO: Make sure the logic here still works when the number of layers is not the same as
    #       the number of trainable layers.
    def _extract_layer_optim_states(self, optimizer_state: PyTree, n_layers: int) -> List[NamedTuple]:
        """Extract the optimizer states for each layer.
        
        The optimizer state is typically given as a named tuple of PyTrees, each PyTree individually
        mimicking the static structure of the model with values for that specific optimization parameter.
        This function breaksthis down into a list of named tuples, each containing the state of each
        parameter for the given layer.
        
        Args:
            optimizer_state: The optax optimizer state to extract the states from.
            n_layers: The number of layers in the model.
            
        Returns:
            A list of named tuples, each containing the state of each parameter for the given layer
        """
        # When there is a tuple of states (chained optimizer),
        # then we just want to the state of the core optimizer
        if type(optimizer_state) == tuple:
            optimizer_state = optimizer_state[0]
        
        # Apply a tree map to the very top level of the optimizer state (each of the different components of the optimizer state).
        # For each of these, if the value is a scalar, then you can just take the scalar.
        # If it is a PyTree, then unzip each layer.
        # From this I should be able to construct a list of states per weight.
        # Then I can apply pass them in the same way I pass in the in/out weights.
        optim_states = jax.tree.map_with_path(
            lambda _, x: (
                [x for _ in range(n_layers)] if jnp.isscalar(x)
                else jax.tree.leaves(x)
            ),
            optimizer_state,
            is_leaf = lambda path, _: len(path) == 1, # Over each comnponent of the optimizer state
            is_leaf_takes_path = True,
        )
        optim_states = [optimizer_state.__class__(*layer_state) for layer_state in zip(*optim_states)]
        return optim_states
    
    def _recombine_layer_optim_states(self, original_optim_state: PyTree, optim_layer_states: List[NamedTuple], ) -> PyTree:
        """Recombine the optimizer states for each layer into a single optimizer state for optax."""
        is_chained = type(original_optim_state) == tuple
        core_optim_state: NamedTuple = original_optim_state[0] if is_chained else original_optim_state
        
        new_optim_state = []
        for i in range(len(core_optim_state)):
            # For scalars, take the value of the scalar in the first layer
            if jnp.isscalar(core_optim_state[i]):
                new_optim_state.append(optim_layer_states[0][i])
            
            # For PyTrees, combine the values from each layer
            else:
                tree_structure = jax.tree.structure(core_optim_state[i])
                new_optim_state.append(
                    jax.tree.map(
                        lambda *args: jax.tree.unflatten(tree_structure, [*args]),
                        *[layer_state[i] for layer_state in optim_layer_states],
                    )
                )
        new_optim_state = core_optim_state.__class__(*new_optim_state)

        if is_chained:
            full_optim_state = (new_optim_state, *original_optim_state[1:])
        else:
            full_optim_state = new_optim_state
        
        return full_optim_state
        
    def prune_features(
        self,
        model: eqx.Module,
        input_values: eqx.Module,
        optimizer: Optional[EqxOptimizer] = None,
        *,
        rng: PRNGKeyArray,
    ) -> Tuple[eqx.Module, EqxOptimizer, List[Bool[Array, 'n_features']]]:
        """Prune features based on CBP utility and return a mask over the features reset.
        
        Args:
            model: The full model to prune
            input_values: Pytree matching the structure of model with the input values for each layer
            optimizer: The optimizer optimizing the given model
            filter_spec: Boolean Pytree matching the structure of model with True for prunable layers
            
        Returns:
            The pruned model, optimizer, and a mask over the features reset
        """
        weights, model_structure = jax.tree.flatten(model)
        optim_layer_states = self._extract_layer_optim_states(optimizer.state, len(weights))
        prune_masks = []
        new_feature_stats = []
        
        # Update from the back to the front
        for i in reversed(range(1, len(weights))):
            rng, layer_rng = jax.random.split(rng)
            
            # Extract values needed for the current layer
            in_weights = weights[i-1] # Shape: (n_features, in_features)
            out_weights = weights[i] # Shape: (out_features, n_features)
            in_optim_state = optim_layer_states[i-1]
            out_optim_state = optim_layer_states[i]
            activation_values = input_values[i] # Shape: (batch_size, n_features)
            feature_stats = self.all_feature_stats[i-1]
            
            # Prune the features
            feature_stats, in_weights, out_weights, in_optim_state, out_optim_state, prune_mask = \
                self.prune_layer_features(
                    in_weights, out_weights, activation_values,
                    feature_stats, in_optim_state, out_optim_state, rng=layer_rng,
                )
            prune_masks.append(prune_mask)
            new_feature_stats.append(feature_stats)
            
            # Apply the updates to the model and optimizer
            weights[i-1] = in_weights
            weights[i] = out_weights
            optim_layer_states[i-1] = in_optim_state
            optim_layer_states[i] = out_optim_state

        # Recombine the weights and optimizer states
        model = jax.tree.unflatten(model_structure, weights)
        new_optim_state = self._recombine_layer_optim_states(optimizer.state, optim_layer_states)
        optimizer = tree_replace(optimizer, state=new_optim_state)
        prune_masks = prune_masks[::-1]
        
        new_tracker = tree_replace(
            self,
            all_feature_stats = new_feature_stats,
            rng = rng,
        )
        
        return new_tracker, model, optimizer, prune_masks
    
    
    def get_statistics(self, layer: eqx.Module):
        pass