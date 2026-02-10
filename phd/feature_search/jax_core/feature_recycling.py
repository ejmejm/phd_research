from functools import partial
import logging
import math
import random
from typing import Dict, List, NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import KeyPath
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from optax import EmptyState

from .models import lecun_uniform
from .optimizers import EqxOptimizer
from .optimizers.adam import AdamState
from .optimizers.idbd import IDBDState
from .utils import get_val_at_key_path, tree_replace, tree_unzip


MAX_FLOAT = jnp.finfo(jnp.float32).max
EPSILON = 1e-8

logger = logging.getLogger(__name__)


class FeatureStats(eqx.Module):
    age: Int[Array, 'n_features']
    utility: Float[Array, 'n_features']
    replacement_accumulator: Float[Array, '']
    init_beta: Float[Array, 'n_features']  # Initial beta when feature was (re)created; used in generate_and_test


class CBPTracker(eqx.Module):
    """Class that performs continual backprop"""
    
    # Static
    incoming_weight_init: str = eqx.field(static=True)
    outgoing_weight_init: str = eqx.field(static=True)
    utility_reset_mode: str = eqx.field(static=True)
    initial_step_size_method: str = eqx.field(static=True)
    maturity_threshold: int = eqx.field(static=True)
    decay_rate: float = eqx.field(static=True)
    init_step_size_lambda: float = eqx.field(static=True)
    init_step_size_gamma: float = eqx.field(static=True)
    protect_growing_step_sizes: bool = eqx.field(static=True)
    prune_eligible_fraction: bool = eqx.field(static=True)

    # Non-static
    replace_rate: float
    all_feature_stats: List[FeatureStats]  # Pytree with FeatureStats for leaves
    global_init_beta: List[Float[Array, '']]  # One scalar per layer; used in generate_and_test
    rng: PRNGKeyArray

    def __init__(
        self,
        model: eqx.Module,
        replace_rate: float = 1e-4,
        decay_rate: float = 0.99,
        maturity_threshold: int = -1,  # -1 means no maturity threshold
        incoming_weight_init: str = 'lecun_uniform',  # {'lecun_uniform', 'kaiming_uniform', 'binary'}
        outgoing_weight_init: str = 'zeros',  # {'zeros', 'lecun_uniform', 'kaiming_uniform'}
        utility_reset_mode: str = 'median',  # {'median', 'zero'}
        initial_step_size_method: str = 'constant',  # {'constant', 'mean', 'median', 'generate_and_test'}
        init_step_size_lambda: float = 1.0,
        init_step_size_gamma: float = 1.0,
        origin_initial_step_size: Optional[float] = None,
        protect_growing_step_sizes: bool = False,
        prune_eligible_fraction: bool = False,
        filter_spec: Optional[PyTree] = None,
        rng: Optional[PRNGKeyArray] = None,
    ):
        assert utility_reset_mode in {'median', 'zero'}
        assert incoming_weight_init in {'lecun_uniform', 'kaiming_uniform', 'binary'}
        assert outgoing_weight_init in {'zeros', 'kaiming_uniform'}
        assert initial_step_size_method in {'constant', 'mean', 'median', 'generate_and_test'}
        if initial_step_size_method == 'generate_and_test':
            assert origin_initial_step_size is not None, "origin_initial_step_size required for generate_and_test"

        origin_init_beta = (
            jnp.log(origin_initial_step_size)
            if initial_step_size_method == 'generate_and_test'
            else jnp.array(0.0, dtype=jnp.float32)
        )

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
        
        # is_linear_weights = lambda x: isinstance(x, Array) and x.ndim == 2 
        assert jnp.all(
            jnp.array(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda x: x.ndim == 2, # Should capture linear weights only
                        eqx.filter(model, lambda x: isinstance(x, Array))
                    )
                )
            )
        ), "All layers must be 2D weight matrices"
        
        weights = jax.tree.leaves(eqx.filter(model, lambda x: isinstance(x, Array)))[1:]
        self.all_feature_stats = [
            FeatureStats(
                age = jnp.zeros(weight_arr.shape[1], dtype=jnp.int32),
                utility = jnp.zeros(weight_arr.shape[1], dtype=jnp.float32),
                replacement_accumulator = jnp.array(0.0, dtype=jnp.float32),
                init_beta = jnp.full(weight_arr.shape[1], origin_init_beta, dtype=jnp.float32),
            )
            for weight_arr in weights
        ]

        self.global_init_beta = [jnp.array(origin_init_beta, dtype=jnp.float32) for _ in weights]
        self.incoming_weight_init = incoming_weight_init
        self.outgoing_weight_init = outgoing_weight_init
        self.utility_reset_mode = utility_reset_mode
        self.initial_step_size_method = initial_step_size_method
        self.maturity_threshold = maturity_threshold
        self.replace_rate = replace_rate
        self.decay_rate = decay_rate
        self.init_step_size_lambda = init_step_size_lambda
        self.init_step_size_gamma = init_step_size_gamma
        self.protect_growing_step_sizes = protect_growing_step_sizes
        self.prune_eligible_fraction = prune_eligible_fraction

        if rng is None:
            rng = jax.random.PRNGKey(random.randint(0, 2**31))
        self.rng = rng
    
    @jax.named_call
    def _compute_step_utility(
        self,
        out_weights: Float[Array, 'out_features in_features'],
        activation_values: Float[Array, 'batch_size in_features'],
        feature_stats: FeatureStats,
        *,
        target: Optional[Float[Array, 'batch_size']] = None,
    ) -> Float[Array, 'in_features']:
        """Compute step utility for each feature. Subclasses override for different utility definitions.

        Returns:
            Step utility array of shape (n_features,) for this step.
        """
        weight_sums = jnp.sum(jnp.abs(out_weights), axis=0)  # (in_features,)
        input_magnitudes = jnp.abs(activation_values).mean(axis=0)  # (in_features,)
        return input_magnitudes * weight_sums

    @jax.named_call
    def _compute_new_feature_stats(
        self,
        feature_stats: FeatureStats,
        weights: Float[Array, 'out_features in_features'],
        input_values: Float[Array, 'batch_size in_features'],
        *,
        target: Optional[Float[Array, 'batch_size']] = None,
    ) -> FeatureStats:
        """Update the feature stats for a single given layer.
        
        Note: replacement_accumulator increment is handled in prune_layer_features
        when prune_eligible_fraction is True, since it requires the full eligibility mask.
        """
        # Age
        age = feature_stats.age + 1

        # Replacement accumulator (only increment here if using total features)
        n_features = weights.shape[1]
        if self.prune_eligible_fraction:
            # Increment handled in prune_layer_features where we have full eligibility
            replacement_accumulator = feature_stats.replacement_accumulator
        else:
            # Increment based on total number of features (original behavior)
            replacement_accumulator = feature_stats.replacement_accumulator + self.replace_rate * n_features

        # Utility
        step_utility = self._compute_step_utility(
            weights, input_values, feature_stats, target=target
        )
        utility = (1 - self.decay_rate) * step_utility + self.decay_rate * feature_stats.utility

        return FeatureStats(
            age = age,
            utility = utility,
            replacement_accumulator = replacement_accumulator,
            init_beta = feature_stats.init_beta,
        )
    
    @partial(jax.jit, static_argnames=('n_replacements',))
    def _make_k_prune_mask(
        self, 
        filtered_utility: Float[Array, 'n_features'],
        eligibility_mask: Bool[Array, 'n_features'],
        n_replacements: int,
    ) -> Bool[Array, 'n_features']:
        """Returns a boolean mask of with approximately n_replacements features to prune.
        
        This is potentially slightly faster than sorting, but it requires knowing exactly
        how many features to prune in advance.
        """
        # lowest to highest utility
        values, _ = jax.lax.top_k(-filtered_utility, n_replacements + 1)
        utility_threshold = -values[-1]
        
        # Construct the prune mask
        prune_mask = jnp.where(filtered_utility < utility_threshold, True, False)
        prune_mask = prune_mask & eligibility_mask
        
        return prune_mask

    @jax.named_call
    def _make_prune_mask(
        self,
        feature_stats: FeatureStats,
        rng: PRNGKeyArray,
        out_optim_state: Optional[NamedTuple] = None,
    ) -> Tuple[Bool[Array, 'n_features'], Int[Array, ''], Int[Array, '']]:
        """Returns a boolean mask of which features to prune and the number of features to prune.
        
        Args:
            feature_stats: Feature statistics including utility and age.
            rng: Random key for tie-breaking.
            out_optim_state: Optional optimizer state for the output weights (used when
                protect_growing_step_sizes is True to exclude features with step-sizes
                greater than their initial step-size).
        
        Returns:
            prune_mask: Boolean mask of which features to prune.
            n_replacements: Number of features to prune this step.
            n_eligible: Number of eligible features (for accumulator increment).
        """
        
        # Determine which features are eligible for replacement
        if self.maturity_threshold > 0:
            eligibility_mask = feature_stats.age > self.maturity_threshold
        else:
            eligibility_mask = jnp.ones(feature_stats.age.shape, dtype=jnp.bool_)
        
        # Exclude features with step-sizes greater than their initial step-size
        if self.protect_growing_step_sizes and out_optim_state is not None:
            if isinstance(out_optim_state, IDBDState):
                # out_optim_state.beta has shape (out_features, n_features)
                # Take mean beta per feature (across output units)
                mean_beta_per_feature = jnp.mean(out_optim_state.beta, axis=0)
                # Compare with init_beta from feature_stats (per-feature)
                step_size_not_growing = mean_beta_per_feature <= feature_stats.init_beta
                eligibility_mask = eligibility_mask & step_size_not_growing
        
        n_eligible = jnp.sum(eligibility_mask)
        
        # Determine number of replacements from accumulator
        n_available_replacements = feature_stats.replacement_accumulator.astype(jnp.int32)
        n_replacements = jnp.minimum(n_available_replacements, n_eligible)
        
        # Compute the threshold for pruning
        # Perturb the utility to avoid ties
        perturbed_utility = feature_stats.utility + \
            jax.random.uniform(rng, feature_stats.utility.shape, minval=-EPSILON, maxval=EPSILON)
        filtered_utility = jnp.where(eligibility_mask, perturbed_utility, jnp.inf)
        utility_ranking = jnp.argsort(filtered_utility)
        utility_threshold = filtered_utility[utility_ranking[n_replacements]]
        # Utilities with inf utility should never be pruned, so set the threshold to the max float
        utility_threshold = jnp.minimum(utility_threshold, MAX_FLOAT)
        
        # Construct the prune mask
        prune_mask = jnp.where(filtered_utility < utility_threshold, True, False)
        prune_mask = prune_mask & eligibility_mask
        
        return prune_mask, n_replacements, n_eligible

    @jax.named_call
    def _update_global_init_beta(
        self,
        feature_stats: FeatureStats,
        global_init_beta: Float[Array, ''],
    ) -> Float[Array, '']:
        """Update global initial beta from utility- and age-weighted average of per-feature init_beta."""
        weights = (self.init_step_size_gamma ** feature_stats.age.astype(jnp.float32)) * feature_stats.utility
        total = jnp.sum(weights) + EPSILON
        weights = weights / total
        target_beta = jnp.sum(feature_stats.init_beta * weights)
        return (
            (1.0 - self.init_step_size_lambda) * global_init_beta
            + self.init_step_size_lambda * target_beta
        ).astype(jnp.float32)
    
    @jax.named_call
    def _reset_feature_stats(
        self,
        feature_stats: FeatureStats,
        prune_mask: Bool[Array, 'n_features'],
        init_beta_after_reset: Optional[Float[Array, 'n_features']] = None,
    ):
        """Resets the feature stats for the given layer and indices."""
        age = jnp.where(prune_mask, 0, feature_stats.age)

        if self.utility_reset_mode == 'median':
            reset_val = jnp.median(feature_stats.utility)
        elif self.utility_reset_mode == 'zero':
            reset_val = 0
        else:
            raise ValueError(f"Invalid utility reset mode: {self.utility_reset_mode}")
        utility = jnp.where(prune_mask, reset_val, feature_stats.utility)

        init_beta = (
            init_beta_after_reset
            if init_beta_after_reset is not None
            else feature_stats.init_beta
        )
        return tree_replace(
            feature_stats,
            age = age,
            utility = utility,
            init_beta = init_beta,
        )
    
    @jax.named_call
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
    
    @jax.named_call
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
    
    @jax.named_call
    def _reset_input_optim_state(
        self,
        optim_layer_state: Optional[NamedTuple],
        prune_mask: Bool[Array, 'n_features'],
        new_init_beta: Optional[Float[Array, 'n_features']] = None,
    ) -> Optional[NamedTuple]:
        """Reset the optimizer state for the weights that output features at the given indices."""
        if optim_layer_state is None:
            return None

        if isinstance(optim_layer_state, IDBDState):
            mean_beta = jnp.mean(optim_layer_state.beta)
            median_beta = jnp.median(optim_layer_state.beta)

        prune_mask_exp = jnp.expand_dims(prune_mask, 1)

        new_vals = []
        for i, value in enumerate(optim_layer_state):
            if value.ndim == 2:
                new_vals.append(jnp.where(prune_mask_exp, 0, value))
            else:
                logger.warning(
                    f"Not resetting optimizer state for field `{optim_layer_state._fields[i]}` because ndim != 2 "
                    f"(not linear weights), ndim: {value.ndim}"
                )
                new_vals.append(value)

        if isinstance(optim_layer_state, IDBDState):
            beta_idx = optim_layer_state._fields.index('beta')
            init_beta_idx = optim_layer_state._fields.index('init_beta')
            if self.initial_step_size_method == 'constant':
                new_vals[beta_idx] = jnp.where(
                    prune_mask_exp, optim_layer_state.init_beta, new_vals[beta_idx]
                )
                # Restore init_beta (was zeroed in loop above)
                new_vals[init_beta_idx] = jnp.where(
                    prune_mask_exp, optim_layer_state.init_beta, new_vals[init_beta_idx]
                )
            elif self.initial_step_size_method == 'mean':
                new_vals[beta_idx] = jnp.where(prune_mask_exp, mean_beta, new_vals[beta_idx])
                # Update init_beta to the actual reset value
                new_vals[init_beta_idx] = jnp.where(prune_mask_exp, mean_beta, new_vals[init_beta_idx])
            elif self.initial_step_size_method == 'median':
                new_vals[beta_idx] = jnp.where(prune_mask_exp, median_beta, new_vals[beta_idx])
                # Update init_beta to the actual reset value
                new_vals[init_beta_idx] = jnp.where(prune_mask_exp, median_beta, new_vals[init_beta_idx])
            elif self.initial_step_size_method == 'generate_and_test' and new_init_beta is not None:
                reset_beta_exp = jnp.expand_dims(new_init_beta, 1)
                new_vals[beta_idx] = jnp.where(
                    prune_mask_exp,
                    reset_beta_exp,
                    new_vals[beta_idx],
                )
                # Update init_beta to the actual reset value
                new_vals[init_beta_idx] = jnp.where(
                    prune_mask_exp,
                    reset_beta_exp,
                    new_vals[init_beta_idx],
                )
            else:
                raise ValueError(
                    f'Invalid initial step-size method: {self.initial_step_size_method}'
                )

        return optim_layer_state.__class__(*new_vals)
    
    @jax.named_call
    def _reset_output_optim_state(
        self,
        optim_layer_state: Optional[NamedTuple],
        prune_mask: Bool[Array, 'n_features'],
        new_init_beta: Optional[Float[Array, 'n_features']] = None,
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

        prune_mask_exp = jnp.expand_dims(prune_mask, 0)

        new_vals = []
        for i, value in enumerate(optim_layer_state):
            if value.ndim == 2:
                new_vals.append(jnp.where(prune_mask_exp, 0, value))
            else:
                logger.warning(
                    f"Not resetting optimizer state for field `{optim_layer_state._fields[i]}` because ndim != 2 "
                    f"(not linear weights), ndim: {value.ndim}"
                )
                new_vals.append(value)

        if isinstance(optim_layer_state, IDBDState):
            beta_idx = optim_layer_state._fields.index('beta')
            init_beta_idx = optim_layer_state._fields.index('init_beta')
            if self.initial_step_size_method == 'constant':
                new_vals[beta_idx] = jnp.where(
                    prune_mask_exp, optim_layer_state.init_beta, new_vals[beta_idx]
                )
                # Restore init_beta (was zeroed in loop above)
                new_vals[init_beta_idx] = jnp.where(
                    prune_mask_exp, optim_layer_state.init_beta, new_vals[init_beta_idx]
                )
            elif self.initial_step_size_method == 'mean':
                new_vals[beta_idx] = jnp.where(prune_mask_exp, mean_betas, new_vals[beta_idx])
                # Update init_beta to the actual reset value
                new_vals[init_beta_idx] = jnp.where(prune_mask_exp, mean_betas, new_vals[init_beta_idx])
            elif self.initial_step_size_method == 'median':
                new_vals[beta_idx] = jnp.where(prune_mask_exp, median_betas, new_vals[beta_idx])
                # Update init_beta to the actual reset value
                new_vals[init_beta_idx] = jnp.where(prune_mask_exp, median_betas, new_vals[init_beta_idx])
            elif self.initial_step_size_method == 'generate_and_test' and new_init_beta is not None:
                reset_beta_exp = jnp.expand_dims(new_init_beta, 0)
                new_vals[beta_idx] = jnp.where(
                    prune_mask_exp,
                    reset_beta_exp,
                    new_vals[beta_idx],
                )
                # Update init_beta to the actual reset value
                new_vals[init_beta_idx] = jnp.where(
                    prune_mask_exp,
                    reset_beta_exp,
                    new_vals[init_beta_idx],
                )
            else:
                raise ValueError(
                    f'Invalid initial step-size method: {self.initial_step_size_method}'
                )

        return optim_layer_state.__class__(*new_vals)
    
    # TODO: Make sure the logic here still works when the number of layers is not the same as
    #       the number of trainable layers.
    def _extract_layer_optim_states(self, optimizer_state: PyTree, key_paths: List[KeyPath]) -> List[NamedTuple]:
        """Extract the optimizer states for each layer.
        
        The optimizer state is typically given as a named tuple of PyTrees, each PyTree individually
        mimicking the static structure of the model with values for that specific optimization parameter.
        This function breaksthis down into a list of named tuples, each containing the state of each
        parameter for the given layer.
        
        Args:
            optimizer_state: The optax optimizer state to extract the states from.
            key_paths: The key paths to the optimizer states.
            
        Returns:
            A list of named tuples, each containing the state of each parameter for the given layer
        """
        # When there is a tuple of states (chained optimizer),
        # then we just want to the state of the core optimizer
        if type(optimizer_state) == tuple:
            optimizer_state = optimizer_state[0]
        
        n_layers = len(key_paths)
        
        if isinstance(optimizer_state, EmptyState):
            return [None for _ in range(n_layers)]
        
        # Apply a tree map to the very top level of the optimizer state (each of the different components of the optimizer state).
        # For each of these, if the value is a scalar, then you can just take the scalar.
        # If it is a PyTree, then unzip each layer.
        # From this I should be able to construct a list of states per weight.
        # Then I can apply pass them in the same way I pass in the in/out weights.
        optim_states = jax.tree.map_with_path(
            lambda _, x: (
                [x for _ in range(n_layers)] if jnp.isscalar(x)
                else [get_val_at_key_path(x, path) for path in key_paths]
            ),
            optimizer_state,
            is_leaf = lambda path, _: len(path) == 1, # Over each component of the optimizer state
            is_leaf_takes_path = True,
        )
        
        # If any of the layers have a None, then the whole state for that layer is None
        optim_states = [
            None if None in layer_state else optimizer_state.__class__(*layer_state)
            for layer_state in zip(*optim_states)
        ]
        return optim_states
    
    
    def _extract_layer_multi_optim_states(
        self, optimizers: Optional[Tuple[EqxOptimizer, ...]], key_paths: List[KeyPath]) -> Tuple[List[NamedTuple], Int[Array, 'n_layers']]:
        """Performs the same as `_extract_layer_optim_states`, but for multiple optimizers.
        
        Extracts separate list of per-layer optimizer states for each optimizer.
        The function combines the these lists into a single list where each layer can only have one corresponding optimizer state.
        The function will error if there are multiple optimizers targeting the same layer.
        
        Args:
            optimizers: The optimizers to extract the states from.
            key_paths: The key paths to the optimizer states.
            
        Returns:
            - A list of named tuples, each containing the optimizer state of the given layer
            - A list of integers mapping the layer index to the index of the optimizer that updates that layer.
              If a layer has no optimizer state, then the value is -1.
        """
        # Extract the optimizer state per layer for each optimizer, then merge them,
        # but keep track of which layer has which state so they can be separated again later
        if optimizers is not None:
            # Get a 2D list of optim states with optimizers in the outer dim, and layers in the inner dim
            per_optim_layer_states = [
                self._extract_layer_optim_states(o.state, key_paths)
                for o in optimizers
            ]
            n_layers = len(per_optim_layer_states[0])
            
            # All optimizers should have states for the same number of layers, even if some are None
            for i in range(1, len(per_optim_layer_states)):
                assert len(per_optim_layer_states[i]) == n_layers, (
                    f"There should be one optimizer state per weight layer, but got "
                    f"{len(per_optim_layer_states[i])} optimizer states and {n_layers} sets of weights!"
                )

            # Create a list that maps the layer index to the index of the optimizer that updates that layer,
            # and combine the per optimizer states into a single list with one state per layer
            layer_optim_mapping = []
            optim_layer_states = []
            for layer_idx in range(n_layers):
                
                optim_idx = -1
                for i, layer_states in enumerate(per_optim_layer_states):
                
                    if layer_states[layer_idx] is not None:
                        if optim_idx == -1: # Good, found the optimizer for this layer  
                            optim_idx = i 
                        else: # Bad, multiple optimizers targeting this layer
                            raise Exception(f"Multiple optimizers targeting layer {layer_idx}!")
                
                if optim_idx == -1: # Bad, no optimizer for this layer
                    logger.warning(
                        f"No optimizer state found for layer {layer_idx}. "
                        "This should only happen if you intend for some layer(s) to be frozen, "
                        "or when you are using a stateless optimizer."
                    )
                    layer_optim_mapping.append(-1)
                    optim_layer_states.append(None)
                else: # Good, found a single optimizer for this layer
                    layer_optim_mapping.append(optim_idx)
                    optim_layer_states.append(per_optim_layer_states[optim_idx][layer_idx])
        
        else:
            optim_layer_states = [None for _ in range(n_layers)]
            layer_optim_mapping = [-1 for _ in range(n_layers)]
        
        return optim_layer_states, layer_optim_mapping
    
    def _recombine_layer_optim_states(
        self, original_optim_state: PyTree,
        optim_layer_states: List[NamedTuple],
    ) -> PyTree:
        """Recombine the optimizer states for each layer into a single optimizer state for optax."""
        is_chained = type(original_optim_state) == tuple
        core_optim_state: NamedTuple = original_optim_state[0] if is_chained else original_optim_state
        
        new_optim_state = []
        
        # Loop through each component of the optimizer state (e.g. `beta`, `h`)
        for i in range(len(core_optim_state)):
            
            # I don't think this will ever be the case, but just to be safe
            if core_optim_state[i] is None:
                new_optim_state.append(None)
            
            # For scalars, take the value of the scalar in the first non-None layer
            elif jnp.isscalar(core_optim_state[i]):
                new_optim_state.append(next(state[i] for state in optim_layer_states if state is not None))
            
            # For PyTrees, combine the values from each layer
            else:
                tree_structure = jax.tree.structure(core_optim_state[i])
                new_optim_state.append(
                    jax.tree.map(
                        lambda *args: jax.tree.unflatten(tree_structure, [*args]),
                        *[layer_state[i] for layer_state in optim_layer_states if layer_state is not None],
                    )
                )
        
        new_optim_state = core_optim_state.__class__(*new_optim_state)

        if is_chained:
            full_optim_state = (new_optim_state, *original_optim_state[1:])
        else:
            full_optim_state = new_optim_state
        
        return full_optim_state
    
    def _recombine_layer_multi_optim_states(
        self,
        optimizers: Optional[Tuple[EqxOptimizer, ...]],
        optim_layer_states: List[NamedTuple],
        layer_optim_mapping: Int[Array, 'n_layers'],
    ) -> Optional[Tuple[EqxOptimizer, ...]]:
        """Recombines the optimizer states for each layer into a single optimizer state for each optimizer."""
        if optimizers is None:
            return None
    
        new_optimizers = []
        for optim_idx, optimizer in enumerate(optimizers):
            # Filter to only get the optimizer states for the layers this optimizer targets
            filtered_optim_layer_states = [
                state if layer_optim_mapping[i] == optim_idx else None
                for i, state in enumerate(optim_layer_states)
            ]
            new_optim_state = self._recombine_layer_optim_states(optimizer.state, filtered_optim_layer_states)
            optimizer = tree_replace(optimizer, state=new_optim_state)
            new_optimizers.append(optimizer)
        return tuple(new_optimizers)

    @jax.named_call
    def prune_layer_features(
        self,
        in_weights: Float[Array, 'n_features in_features'],
        out_weights: Float[Array, 'out_features n_features'],
        activation_values: Float[Array, 'batch_size n_features'],
        feature_stats: FeatureStats,
        layer_global_init_beta: Float[Array, ''],
        in_optim_state: Optional[NamedTuple] = None,
        out_optim_state: Optional[NamedTuple] = None,
        *,
        rng: PRNGKeyArray,
        target: Optional[Float[Array, 'batch_size']] = None,
    ) -> Tuple[FeatureStats, Float[Array, 'n_features in_features'], Float[Array, 'out_features n_features'], Optional[NamedTuple], Optional[NamedTuple], Bool[Array, 'n_features'], Int[Array, ''], Int[Array, ''], Float[Array, '']]:
        assert in_weights.ndim == 2, "Weights must be 2D"
        assert out_weights.ndim == 2, "Weights must be 2D"

        in_weight_key, out_weight_key, prune_mask_key, sample_key = jax.random.split(rng, 4)

        # Update feature stats
        feature_stats = self._compute_new_feature_stats(
            feature_stats, out_weights, activation_values, target=target
        )

        # Update global init beta (generate_and_test): pull towards init_beta of high-utility, recent features
        updated_layer_global_init_beta = jnp.where(
            self.initial_step_size_method == 'generate_and_test',
            self._update_global_init_beta(feature_stats, layer_global_init_beta),
            layer_global_init_beta,
        )

        # Get indices to reinitialize (prune mask)
        prune_mask, n_replacements, n_eligible = self._make_prune_mask(
            feature_stats, prune_mask_key, out_optim_state
        )

        # Fraction of pruned features with age > 0.5 / recycle_rate (long-lived)
        age_threshold = 0.5 / self.replace_rate
        long_lived_mask = (feature_stats.age.astype(jnp.float32) > age_threshold) & prune_mask
        n_long_lived = jnp.sum(long_lived_mask)
        n_pruned_layer = jnp.sum(prune_mask)

        # Update replacement accumulator
        if self.prune_eligible_fraction:
            # Increment based on eligible features (full eligibility mask)
            accumulator_increment = self.replace_rate * n_eligible.astype(jnp.float32)
            new_accumulator = feature_stats.replacement_accumulator + accumulator_increment - n_replacements
        else:
            # Increment already done in _compute_new_feature_stats, just decrement
            new_accumulator = feature_stats.replacement_accumulator - n_replacements
        feature_stats = tree_replace(
            feature_stats,
            replacement_accumulator = new_accumulator,
        )

        # Compute the reset beta value based on the initial_step_size_method
        n_features = feature_stats.utility.shape[0]
        if self.initial_step_size_method == 'generate_and_test':
            # Sample new init_beta for pruned features: uniform in [global - 0.5, global + 0.5]
            reset_beta_value = jax.random.uniform(
                sample_key,
                (n_features,),
                minval = updated_layer_global_init_beta - 0.5,
                maxval = updated_layer_global_init_beta + 0.5,
                dtype = jnp.float32,
            )
        elif self.initial_step_size_method == 'median' and out_optim_state is not None:
            if isinstance(out_optim_state, IDBDState):
                reset_beta_value = jnp.full(
                    n_features, jnp.median(out_optim_state.beta), dtype=jnp.float32
                )
            else:
                reset_beta_value = feature_stats.init_beta
        elif self.initial_step_size_method == 'mean' and out_optim_state is not None:
            if isinstance(out_optim_state, IDBDState):
                reset_beta_value = jnp.full(
                    n_features, jnp.mean(out_optim_state.beta), dtype=jnp.float32
                )
            else:
                reset_beta_value = feature_stats.init_beta
        elif self.initial_step_size_method == 'constant' and out_optim_state is not None:
            if isinstance(out_optim_state, IDBDState):
                # Use mean of out_optim_state.init_beta as representative value
                reset_beta_value = jnp.full(
                    n_features, jnp.mean(out_optim_state.init_beta), dtype=jnp.float32
                )
            else:
                reset_beta_value = feature_stats.init_beta
        else:
            reset_beta_value = feature_stats.init_beta

        init_beta_after_reset = jnp.where(
            prune_mask,
            reset_beta_value,
            feature_stats.init_beta,
        )

        # Reset stats for those features (update init_beta to actual reset value)
        feature_stats = self._reset_feature_stats(
            feature_stats, prune_mask, init_beta_after_reset=init_beta_after_reset
        )

        # Reinit input and output weights for given features
        in_weights = self._reinit_input_weights(in_weights, prune_mask, in_weight_key)
        out_weights = self._reinit_output_weights(out_weights, prune_mask, out_weight_key)

        # Reinit optimizer input and output weight states for given features
        new_init_beta_for_optim = (
            init_beta_after_reset if self.initial_step_size_method == 'generate_and_test' else None
        )
        in_optim_state = self._reset_input_optim_state(
            in_optim_state, prune_mask, new_init_beta=new_init_beta_for_optim
        )
        out_optim_state = self._reset_output_optim_state(
            out_optim_state, prune_mask, new_init_beta=new_init_beta_for_optim
        )

        return (
            feature_stats,
            in_weights,
            out_weights,
            in_optim_state,
            out_optim_state,
            prune_mask,
            n_long_lived,
            n_pruned_layer,
            updated_layer_global_init_beta,
        )
        
    def prune_features(
        self,
        model: eqx.Module,
        input_values: eqx.Module,
        optimizers: EqxOptimizer | Tuple[EqxOptimizer, ...] | None = None,
        *,
        rng: PRNGKeyArray,
        targets: Optional[Float[Array, 'batch_size']] = None,
    ) -> Tuple['CBPTracker', eqx.Module, EqxOptimizer, List[Bool[Array, 'n_features']], Float[Array, '']]:
        """Prune features based on the CBP score.

        Args:
            model: The full model to prune
            input_values: Pytree matching the structure of model with the input values for each layer
            optimizer: The optimizer optimizing the given model
            filter_spec: Boolean Pytree matching the structure of model with True for prunable layers

        Returns:
            The pruned model, optimizer, and a mask over the features reset
        """
        if isinstance(optimizers, EqxOptimizer):
            optimizers = (optimizers,)
            single_optimizer = True
        else:
            single_optimizer = False

        leaves, model_structure = jax.tree.flatten_with_path(model)
        weight_paths, weights = zip(*leaves)
        weights = list(weights)
        optim_layer_states, layer_optim_mapping = self._extract_layer_multi_optim_states(
            optimizers, weight_paths
        )
        prune_masks = []
        new_feature_stats = []
        new_global_init_beta = list(self.global_init_beta)
        total_n_long_lived = 0
        total_n_pruned = 0
        indices = list(reversed(range(1, len(weights))))

        # Update from the back to the front
        for idx, i in enumerate(indices):
            rng, layer_rng = jax.random.split(rng)

            # Extract values needed for the current layer
            in_weights = weights[i - 1]  # Shape: (n_features, in_features)
            out_weights = weights[i]  # Shape: (out_features, n_features)
            in_optim_state = optim_layer_states[i - 1]
            out_optim_state = optim_layer_states[i]
            activation_values = input_values[i]  # Shape: (batch_size, n_features)
            feature_stats = self.all_feature_stats[i - 1]
            layer_global_init_beta = self.global_init_beta[i - 1]

            layer_target = None
            if targets is not None and idx == 0:
                layer_target = jnp.squeeze(targets, axis=-1) if targets.ndim > 1 else targets

            # Prune the features
            (
                feature_stats,
                in_weights,
                out_weights,
                in_optim_state,
                out_optim_state,
                prune_mask,
                n_long_lived,
                n_pruned_layer,
                updated_layer_global_init_beta,
            ) = self.prune_layer_features(
                in_weights,
                out_weights,
                activation_values,
                feature_stats,
                layer_global_init_beta,
                in_optim_state,
                out_optim_state,
                rng=layer_rng,
                target=layer_target,
            )
            new_global_init_beta[i - 1] = updated_layer_global_init_beta
            prune_masks.append(prune_mask)
            new_feature_stats.append(feature_stats)
            total_n_long_lived = total_n_long_lived + n_long_lived
            total_n_pruned = total_n_pruned + n_pruned_layer

            # Apply the updates to the model and optimizer
            weights[i-1] = in_weights
            weights[i] = out_weights
            optim_layer_states[i-1] = in_optim_state
            optim_layer_states[i] = out_optim_state

        # Recombine the weights and optimizer states
        model = jax.tree.unflatten(model_structure, weights)
        optimizers = self._recombine_layer_multi_optim_states(
            optimizers, optim_layer_states, layer_optim_mapping)
        
        prune_masks = prune_masks[::-1]

        long_lived_frac = (total_n_long_lived.astype(jnp.float32) /
                          jnp.maximum(total_n_pruned.astype(jnp.float32), 1.0))

        new_tracker = tree_replace(
            self,
            all_feature_stats = new_feature_stats,
            global_init_beta = new_global_init_beta,
            rng = rng,
        )
        
        if single_optimizer:
            optimizers = optimizers[0]
        
        return new_tracker, model, optimizers, prune_masks, long_lived_frac
    
    def update_feature_stats(
        self,
        model: eqx.Module,
        input_values: eqx.Module,
        targets: Optional[Float[Array, 'batch_size']] = None,
    ) -> 'CBPTracker':
        """Updates the feature stats based on activation values and returns an updated tracker.

        Args:
            model: The full model to update the feature stats for
            input_values: Pytree matching the structure of model with the input values for each layer

        Returns:
            The updated tracker
        """
        weights = jax.tree.leaves(model)
        new_feature_stats = []
        new_global_init_beta = list(self.global_init_beta)
        indices = list(reversed(range(1, len(weights))))

        # Update from the back to the front
        for idx, i in enumerate(indices):
            # Extract values needed for the current layer
            in_weights = weights[i - 1]  # Shape: (n_features, in_features)
            out_weights = weights[i]  # Shape: (out_features, n_features)
            activation_values = input_values[i]  # Shape: (batch_size, n_features)
            feature_stats = self.all_feature_stats[i - 1]

            assert in_weights.ndim == 2, "Weights must be 2D"
            assert out_weights.ndim == 2, "Weights must be 2D"

            layer_target = None
            if targets is not None and idx == 0:
                layer_target = jnp.squeeze(targets, axis=-1) if targets.ndim > 1 else targets
            # Update feature stats
            feature_stats = self._compute_new_feature_stats(
                feature_stats, out_weights, activation_values, target=layer_target
            )
            # Update global init beta (generate_and_test)
            new_global_init_beta[i - 1] = jnp.where(
                self.initial_step_size_method == 'generate_and_test',
                self._update_global_init_beta(feature_stats, self.global_init_beta[i - 1]),
                self.global_init_beta[i - 1],
            )
            new_feature_stats.append(feature_stats)

            # Apply the updates to the model and optimizer
            weights[i - 1] = in_weights
            weights[i] = out_weights

        new_tracker = tree_replace(
            self,
            all_feature_stats = new_feature_stats,
            global_init_beta = new_global_init_beta,
        )

        return new_tracker
    
    
    def get_statistics(self, layer: eqx.Module):
        pass


class SignedCBPTracker(CBPTracker):
    """CBP tracker that uses signed utility: |error + contribution| - |error| per feature.

    We prune the lowest-utility features. Requires targets to be passed to
    prune_features and update_feature_stats so the output layer receives them.
    """

    @jax.named_call
    def _compute_step_utility(
        self,
        out_weights: Float[Array, 'out_features in_features'],
        activation_values: Float[Array, 'batch_size in_features'],
        feature_stats: FeatureStats,
        *,
        target: Optional[Float[Array, 'batch_size']] = None,
    ) -> Float[Array, 'in_features']:
        """Signed utility: mean over batch of |error + contribution_j| - |error|."""
        if target is None:
            return super()._compute_step_utility(
                out_weights, activation_values, feature_stats, target=target
            )
        contributions = activation_values * out_weights[0]  # (batch_size, n_features)
        pred = jnp.sum(contributions, axis=1)
        target_error = target - pred
        return jnp.mean(
            jnp.abs(target_error[:, None] + contributions)
            - jnp.abs(target_error)[:, None],
            axis=0,
        )