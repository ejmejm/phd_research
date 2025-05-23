from typing import Tuple, Optional, Iterator, Dict, Any, Callable, List

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random, lax

from ..utils import tree_replace


class LTU(eqx.Module):
    """Linear Threshold Unit activation function."""
    
    def __call__(self, x):
        return jnp.where(x > 0, jnp.ones_like(x), jnp.zeros_like(x))


class NonlinearGEOFFTask(eqx.Module):
    """Non-linear version of GEOFF task with configurable depth and activation.
    
    This implements the JAX version of the GEOFF task for use with Equinox.
    Model is structured as an Equinox module with separate static and non-static
    parameters as needed.
    """
    
    # Static parameters (configuration)
    n_features: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    n_stationary_layers: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    weight_scale: float = eqx.field(static=True)
    flip_rate: float = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    sparsity: float = eqx.field(static=True)
    weight_init: str = eqx.field(static=True)
    standard_input: bool = eqx.field(static=True)
    
    # Dynamic parameters (weights and state)
    weights: List[jax.Array]
    flip_accumulators: List[float]
    input_mean: Optional[jax.Array] = None
    input_std: Optional[jax.Array] = None
    rng: random.PRNGKey

    def __init__(
        self,
        n_features: int,
        flip_rate: float,  # Percentage of weights to flip per step
        n_layers: int = 2,
        n_stationary_layers: int = 0,
        hidden_dim: int = 64,
        weight_scale: float = 1.0,
        activation: str = 'relu',
        sparsity: float = 0.0,
        weight_init: str = 'binary',
        input_mean_range: Tuple[float, float] = (0, 0),
        input_std_range: Tuple[float, float] = (1, 1),
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_features: Number of input features
            flip_rate: Percentage of weights to flip per step (accumulates if < 1 weight)
            n_layers: Number of layers in the target network (1 = linear)
            n_stationary_layers: Number of layers that do not flip
            hidden_dim: Hidden dimension size for intermediate layers
            weight_scale: Scale factor for weights (weights will be ±scale)
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            sparsity: Percentage of weights (other than the last layer) to set to zero
            weight_init: Weight initialization method ('binary' or 'kaiming_uniform')
            input_mean_range: Range for random input mean values
            input_std_range: Range for random input std values
            seed: Random seed for reproducibility
        """
        assert weight_init in ['binary', 'kaiming_uniform'], f"Unsupported weight initialization: {weight_init}"
        
        super().__init__()
        
        # Store static configuration
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_stationary_layers = n_stationary_layers
        self.hidden_dim = hidden_dim
        self.weight_scale = weight_scale
        self.flip_rate = flip_rate
        self.activation = activation
        self.sparsity = sparsity
        self.weight_init = weight_init
        
        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 1000000000)
        key = random.PRNGKey(seed)
        
        # Handle input distribution
        self.standard_input = (
            tuple(input_mean_range) == (0, 0) and 
            tuple(input_std_range) == (1, 1)
        )
        
        if not self.standard_input:
            # Generate uniform samples for mean and std
            mean_key, std_key, key = random.split(key, 3)
            self.input_mean = random.uniform(
                mean_key, 
                shape=(n_features,), 
                minval=input_mean_range[0], 
                maxval=input_mean_range[1]
            )
            self.input_std = random.uniform(
                std_key, 
                shape=(n_features,), 
                minval=input_std_range[0], 
                maxval=input_std_range[1]
            )
        
        # Initialize network weights and accumulators based on layers configuration
        if n_layers == 1:
            # Single linear layer
            weight_key, key = random.split(key)
            layer_weights = self._initialize_weights(weight_key, n_features, 1)
            self.weights = [layer_weights]
            self.flip_accumulators = [flip_rate * n_features]
        else:
            # Multiple layers with hidden dimensions
            # We'll use JAX's functional approach to build the weights list
            keys = random.split(key, 2 * n_layers - 1)
            weight_keys = keys[:n_layers]
            sparsify_keys = keys[n_layers:]
            
            # Prepare dimensions for each layer
            in_dims = [n_features] + [hidden_dim] * (n_layers - 1)
            out_dims = [hidden_dim] * (n_layers - 1) + [1]
            
            # Initialize weights for each layer
            all_weights = []
            all_accumulators = []
            
            # Input layer
            w = self._initialize_weights(weight_keys[0], in_dims[0], out_dims[0])
            w = self._sparsify_weights(sparsify_keys[0], w, sparsity)
            all_weights.append(w)
            all_accumulators.append(flip_rate * in_dims[0] * out_dims[0])
            
            # Hidden layers
            for i in range(1, n_layers - 1):
                w = self._initialize_weights(weight_keys[i], in_dims[i], out_dims[i])
                w = self._sparsify_weights(sparsify_keys[i], w, sparsity)
                all_weights.append(w)
                all_accumulators.append(flip_rate * in_dims[i] * out_dims[i])
            
            # Output layer
            w = self._initialize_weights(weight_keys[-1], in_dims[-1], out_dims[-1])
            all_weights.append(w)
            all_accumulators.append(flip_rate * in_dims[-1] * out_dims[-1])
            
            self.weights = all_weights
            self.flip_accumulators = all_accumulators

        self.rng = key
    
    def _initialize_weights(self, key: random.PRNGKey, in_features: int, out_features: int) -> jax.Array:
        """Initialize weights based on specified initialization method."""
        if self.weight_init == 'binary':
            # Create binary ±1 weights
            weights = random.randint(key, (in_features, out_features), 0, 2) * 2 - 1
            weights = weights.astype(jnp.float32)
        else:  # kaiming_uniform
            # Using He initialization (kaiming)
            limit = jnp.sqrt(6 / in_features)  # He/kaiming initialization
            weights = random.uniform(key, (in_features, out_features), -limit, limit)
            
        return weights * self.weight_scale
    
    def _sparsify_weights(self, key: random.PRNGKey, weights: jax.Array, sparsity: float) -> jax.Array:
        """Set a percentage of weights to zero based on random mask.
        
        Args:
            key: PRNG key
            weights: Weight matrix to sparsify
            sparsity: Fraction of weights to set to zero (0.0 to 1.0)
            
        Returns:
            Sparsified weight matrix
        """
        if sparsity == 0:
            return weights
            
        # Create random mask
        mask = random.uniform(key, weights.shape) >= sparsity
        return weights * mask
    
    def _get_activation_fn(self, x: jax.Array) -> jax.Array:
        """Apply the configured activation function to input."""
        # Use pattern matching to select activation function
        if self.activation == 'relu':
            return jax.nn.relu(x)
        elif self.activation == 'tanh':
            return jnp.tanh(x)
        elif self.activation == 'sigmoid':
            return jax.nn.sigmoid(x)
        elif self.activation == 'ltu':
            return jnp.where(x > 0, jnp.ones_like(x), jnp.zeros_like(x))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def _forward(self, x: jax.Array) -> jax.Array:
        """Forward pass through the target network."""
        # Handle linear case specially
        if self.n_layers == 1:
            return x @ self.weights[0]
        
        for i in range(self.n_layers - 1):
            x = x @ self.weights[i]
            x = self._get_activation_fn(x)
        
        return x @ self.weights[-1]
    
    def _flip_signs(self, key: random.PRNGKey) -> Tuple[list, list, random.PRNGKey]:
        """Flip signs of weights based on accumulated probabilities."""
        # Make a mutable copy of the weights and accumulators
        new_weights = [w.copy() for w in self.weights]
        new_accumulators = [a for a in self.flip_accumulators]
        
        for layer_idx in range(self.n_stationary_layers, len(self.weights)):
            weights = new_weights[layer_idx]
            accumulator = new_accumulators[layer_idx]
            
            # Use a fixed-trace approach with a percentage mask instead of random.choice
            def flip_weights(weights, accumulator, key):
                # Get total number of elements in the weight matrix
                total_elements = weights.size
                n_flips = jnp.floor(accumulator).astype(jnp.int32)
                
                # Create a random mask with exactly n_flips ones
                # Use uniform random values and get the top-n_flips values
                key, subkey = random.split(key)
                random_vals = random.uniform(subkey, shape=(total_elements,))
                
                # Create a boolean mask for the indices with top-n_flips values
                threshold = jnp.sort(random_vals)[-n_flips]
                mask = (random_vals >= threshold).astype(jnp.int32)
                
                # Ensure we have exactly n_flips ones (handling ties)
                # This is needed because ties in random values might give more than n_flips ones
                cumsum = jnp.cumsum(mask)
                mask = jnp.where(cumsum <= n_flips, mask, 0)
                
                # Reshape weights to 1D for indexing
                flat_weights = weights.reshape(-1)
                
                # Apply flip using the mask (-1 where mask is 1, 1 otherwise)
                flip_factors = 1 - 2 * mask
                flipped_weights = flat_weights * flip_factors
                
                # Return reshaped weights and updated accumulator
                return flipped_weights.reshape(weights.shape), accumulator - n_flips, key
                
            # Use lax.cond for conditional execution
            n_flips = jnp.floor(accumulator)
            new_weights[layer_idx], new_accumulators[layer_idx], key = lax.cond(
                n_flips > 0,
                lambda k: flip_weights(weights, accumulator, k),
                lambda k: (weights, accumulator, k),
                key
            )
        return new_weights, new_accumulators, key
    
    def generate_batch(self, batch_size: int = 1) -> Tuple[eqx.Module, Tuple]:
        """Generates a single batch of data.
        
        Args:
            batch_size: Size of batch to generate
            
        Returns:
            Tuple containing:
            - New task state
            - Batch data (x, y)
        """
        # Accumulate and handle weight flips
        new_accumulators = []
        for i in range(len(self.flip_accumulators)):
            if self.n_layers == 1:
                n_flippable = self.n_features
            elif i == 0:
                n_flippable = self.n_features * self.hidden_dim
            elif i == len(self.weights) - 1:
                n_flippable = self.hidden_dim
            else:
                n_flippable = self.hidden_dim * self.hidden_dim
                
            new_accumulators.append(self.flip_accumulators[i] + self.flip_rate * n_flippable)
        
        new_rng, flip_key, x_key = random.split(self.rng, 3)

        # Flip weights according to accumulators
        new_weights, new_accumulators, key = self._flip_signs(key=flip_key)
        
        # Generate random input features
        x = random.normal(x_key, (batch_size, self.n_features))
        
        # Apply custom input distribution if needed
        if not self.standard_input:
            x = x * self.input_std + self.input_mean
        
        # Create a temporary task with the updated weights for forward pass
        new_task_state = tree_replace(
            self,
            weights = new_weights,
            flip_accumulators = new_accumulators,
            rng = new_rng
        )
        
        # Forward pass through target network
        y = jax.vmap(new_task_state._forward)(x)
        
        # Return updated state and the batch
        return new_task_state, (x, y)