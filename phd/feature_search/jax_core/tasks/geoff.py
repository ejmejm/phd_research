from typing import Tuple, Optional, Iterator, Dict, Any, Callable, List

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random, lax
from jaxtyping import Array, Float, Int

from ..utils import tree_replace


def initialize_weights(
    key: random.PRNGKey,
    in_features: int,
    out_features: int,
    weight_init: str = 'binary',
    weight_scale: float = 1.0,
) -> jax.Array:
    """Initialize weights based on specified initialization method.
    
    Args:
        key: PRNG key for random initialization
        in_features: Number of input features
        out_features: Number of output features
        weight_init: Initialization method ('binary' or 'kaiming_uniform')
        weight_scale: Scale factor for weights
        
    Returns:
        Initialized weight matrix of shape (in_features, out_features)
    """
    if weight_init == 'binary':
        weights = random.randint(key, (in_features, out_features), 0, 2) * 2 - 1
        weights = weights.astype(jnp.float32)
    elif weight_init == 'kaiming_uniform':
        limit = jnp.sqrt(6 / in_features)
        weights = random.uniform(key, (in_features, out_features), jnp.float32, -limit, limit)
    else:
        raise ValueError(f"Unsupported weight initialization: {weight_init}")
    
    return weights * weight_scale


def sparsify_weights(
    key: random.PRNGKey,
    weights: jax.Array,
    sparsity: float,
) -> jax.Array:
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
    
    mask = random.uniform(key, weights.shape) >= sparsity
    return weights * mask


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
    n_outputs: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    n_stationary_layers: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    weight_scale: float = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    sparsity: float = eqx.field(static=True)
    weight_init: str = eqx.field(static=True)
    standard_input: bool = eqx.field(static=True)
    
    # Dynamic parameters (weights and state)
    weights: List[Float[Array, 'in_features out_features']]
    flip_accumulators: Float[Array, 'n_layers']
    _n_flippable: Int[Array, 'n_layers']
    flip_rate: Float[Array, 'n_layers']
    input_mean: Optional[Float[Array, 'n_features']] = None
    input_std: Optional[Float[Array, 'n_features']] = None
    rng: random.PRNGKey

    def __init__(
        self,
        n_features: int,
        flip_rate: float,  # Percentage of weights to flip per step
        n_layers: int = 2,
        n_stationary_layers: int = 0,
        hidden_dim: int = 64,
        n_outputs: int = 1,
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
            n_outputs: Number of output dimensions
            weight_scale: Scale factor for weights (weights will be ±scale)
            activation: Activation function ('ltu', 'relu', 'tanh', or 'sigmoid')
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
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_stationary_layers = n_stationary_layers
        self.hidden_dim = hidden_dim
        self.weight_scale = weight_scale
        self.flip_rate = jnp.array(flip_rate, dtype=jnp.float32)
        self.activation = activation
        self.sparsity = sparsity
        self.weight_init = weight_init
        
        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 2**31)
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
                shape = (n_features,),
                minval = input_mean_range[0],
                maxval = input_mean_range[1],
            )
            self.input_std = random.uniform(
                std_key,
                shape = (n_features,),
                minval = input_std_range[0],
                maxval = input_std_range[1],
            )
        
        key, network_key = random.split(key)
        self._initialize_network(
            n_layers = n_layers,
            n_features = n_features,
            hidden_dim = hidden_dim,
            n_outputs = n_outputs,
            flip_rate = flip_rate,
            sparsity = sparsity,
            key = network_key,
        )
        
        n_flippable = []
        for i in range(len(self.flip_accumulators)):
            if self.n_layers == 1:
                n_flippable = self.n_features * self.n_outputs
            elif i == 0:
                n_flippable = self.n_features * self.hidden_dim
            elif i == len(self.weights) - 1:
                n_flippable = self.hidden_dim * self.n_outputs
            else:
                n_flippable = self.hidden_dim * self.hidden_dim
        self._n_flippable = jnp.array(n_flippable, dtype=jnp.int32)

        self.rng = key
        
    def _initialize_network(
        self, n_layers: int, n_features: int, hidden_dim: int, n_outputs: int,
        flip_rate: float, sparsity: float, key: random.PRNGKey,
    ):
        # Initialize network weights and accumulators based on layers configuration
        if n_layers == 1:
            # Single linear layer
            weight_key, key = random.split(key)
            layer_weights = initialize_weights(
                weight_key, n_features, n_outputs,
                weight_init = self.weight_init,
                weight_scale = self.weight_scale,
            )
            self.weights = [layer_weights]
            flip_accumulators = [flip_rate * n_features * n_outputs]
        else:
            # Multiple layers with hidden dimensions
            keys = random.split(key, 2 * n_layers - 1)
            weight_keys = keys[:n_layers]
            sparsify_keys = keys[n_layers:]
            
            # Prepare dimensions for each layer
            in_dims = [n_features] + [hidden_dim] * (n_layers - 1)
            out_dims = [hidden_dim] * (n_layers - 1) + [n_outputs]
            
            # Initialize weights for each layer
            all_weights = []
            all_accumulators = []
            
            # Input layer
            w = initialize_weights(
                weight_keys[0], in_dims[0], out_dims[0],
                weight_init = self.weight_init,
                weight_scale = self.weight_scale,
            )
            w = sparsify_weights(sparsify_keys[0], w, sparsity)
            all_weights.append(w)
            all_accumulators.append(flip_rate * in_dims[0] * out_dims[0])
            
            # Hidden layers
            for i in range(1, n_layers - 1):
                w = initialize_weights(
                    weight_keys[i], in_dims[i], out_dims[i],
                    weight_init = self.weight_init,
                    weight_scale = self.weight_scale,
                )
                w = sparsify_weights(sparsify_keys[i], w, sparsity)
                all_weights.append(w)
                all_accumulators.append(flip_rate * in_dims[i] * out_dims[i])
            
            # Output layer
            w = initialize_weights(
                weight_keys[-1], in_dims[-1], out_dims[-1],
                weight_init = self.weight_init,
                weight_scale = self.weight_scale,
            )
            all_weights.append(w)
            all_accumulators.append(flip_rate * in_dims[-1] * out_dims[-1])
            
            self.weights = all_weights
            flip_accumulators = all_accumulators
            
        self.flip_accumulators = jnp.array(flip_accumulators, dtype=jnp.float32)
    
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
    
    # TODO: Change this to account for sparsity
    def _flip_signs(
        self,
        weights: Float[Array, 'in_features out_features'],
        flip_accumulator: Float[Array, ''],
        key: random.PRNGKey,
    ) -> Tuple[Float[Array, 'in_features out_features'], Float[Array, '']]:
        """Flips the signs of weights based on the flip accumulator.
        
        Args:
            weights: Weight matrix to flip signs of
            flip_accumulator: Flip accumulator for the layer
            key: PRNG key
            
        Returns:
            Tuple containing:
            - Flipped weight matrix
            - Updated flip accumulator
        """
        n_flips = jnp.floor(flip_accumulator).astype(jnp.int32)
        flip_accumulator = flip_accumulator - n_flips
        
        # Randomly select weights to flip
        flat_idx = random.permutation(key, weights.size)
        weights = weights * jnp.where(flat_idx < n_flips, -1, 1).reshape(*weights.shape)
        
        return weights, flip_accumulator

    def generate_batch(self, batch_size: int = 1) -> Tuple[eqx.Module, Tuple]:
        """Generates a single batch of data.
        
        Args:
            batch_size: Size of batch to generate
            
        Returns:
            Tuple containing:
            - New task state
            - Batch data (x, y)
        """
        accumulators = self.flip_accumulators + self.flip_rate * self._n_flippable
        
        new_rng, flip_key, x_key = random.split(self.rng, 3)

        # Flip weights according to accumulators
        new_weights = [self.weights[i] for i in range(self.n_stationary_layers)]
        new_accumulators = [0.0 for _ in range(self.n_stationary_layers)]
        flip_keys = random.split(flip_key, len(self.weights))
        for layer_idx in range(self.n_stationary_layers, len(self.weights)):
            new_weight, accumulator = self._flip_signs(
                weights = self.weights[layer_idx],
                flip_accumulator = accumulators[layer_idx],
                key = flip_keys[layer_idx],
            )
            new_weights.append(new_weight)
            new_accumulators.append(accumulator)
        new_accumulators = jnp.array(new_accumulators, dtype=jnp.float32)
        
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


class InputChangingGEOFFTask(NonlinearGEOFFTask):
    """Input changing version of GEOFF task with configurable depth and activation.
    
    This implements the JAX version of the GEOFF task for use with Equinox.
    Model is structured as an Equinox module with separate static and non-static
    parameters as needed.
    """
    # Input distribution parameters
    input_bounds: Tuple[float, float] = eqx.field(static=True) # Overall bounds of the input space
    input_change_freq: Optional[int] = eqx.field(static=True) # Number of steps between input changes
    input_subspace_range: float = eqx.field(static=True) # Range of the uniform distributions for sampling input values
    max_input_center_change: float = eqx.field(static=True) # Maximum change of a subspace center per change step
    
    bias: Float[Array, ''] # Bias unit that is not used by default
    input_subspace_centers: Float[Array, 'n_features'] # Centers of the uniform distributions for sampling input values
    step: Int[Array, ''] # Current step

    def __init__(
        self,
        n_features: int,
        flip_rate: float, # Percentage of weights to flip per step
        n_layers: int = 2,
        n_stationary_layers: int = 0,
        hidden_dim: int = 64,
        n_outputs: int = 1,
        weight_scale: float = 1.0,
        activation: str = 'relu',
        sparsity: float = 0.0,
        weight_init: str = 'binary',
        input_bounds: Tuple[float, float] = (-1.0, 1.0),
        input_subspace_range: float = 0.1,
        input_change_freq: Optional[int] = None, # Int unless inf
        max_input_center_change: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_features: Number of input features
            flip_rate: Percentage of weights to flip per step (accumulates if < 1 weight)
            n_layers: Number of layers in the target network (1 = linear)
            n_stationary_layers: Number of layers that do not flip
            hidden_dim: Hidden dimension size for intermediate layers
            n_outputs: Number of output dimensions
            weight_scale: Scale factor for weights (weights will be ±scale)
            activation: Activation function ('ltu', 'relu', 'tanh', or 'sigmoid')
            sparsity: Percentage of weights (other than the last layer) to set to zero
            weight_init: Weight initialization method ('binary' or 'kaiming_uniform')
            input_bounds: Overall bounds of the input space
            input_subspace_range: Range of the uniform distributions for sampling input values
            input_change_freq: Number of steps between input changes (None if it doesn't change)
            max_input_center_change: Maximum change of a subspace center per change step
            seed: Random seed for reproducibility
        """
        super().__init__(
            n_features, flip_rate, n_layers, n_stationary_layers, hidden_dim,
            n_outputs, weight_scale, activation, sparsity, weight_init, (0, 0), (1, 1), seed,
        )
        self.rng, input_distrib_key = random.split(self.rng, 2)
        
        self.standard_input = False
        self.input_mean = None
        self.input_std = None
        self.bias = jnp.array(0.0, dtype=jnp.float32)
        
        # Initialize input distribution params
        self.input_change_freq = input_change_freq
        self.input_bounds = input_bounds
        self.input_subspace_range = input_subspace_range
        self.max_input_center_change = max_input_center_change
        self.input_subspace_centers = jax.random.uniform(
            input_distrib_key, (n_features,), jnp.float32, self.input_bounds[0], self.input_bounds[1])
        self.step = jnp.array(0, dtype=jnp.int32)
    
    def _compute_updated_input_subspace_centers(self, rng: random.PRNGKey) -> Float[Array, 'n_features']:
        min_change = -self.max_input_center_change
        max_change = self.max_input_center_change
        center_shifts = jax.random.uniform(rng, (self.n_features,), jnp.float32, min_change, max_change)
        new_centers = self.input_subspace_centers + center_shifts
        min_bound, max_bound = self.input_bounds
        range_size = max_bound - min_bound
        new_centers = min_bound + jnp.mod(new_centers - min_bound, range_size)
        return new_centers
    
    def _sample_inputs(self, rng: random.PRNGKey, batch_size: int = 1) -> Tuple[Float[Array, 'batch_size n_features']]:
        bound = self.input_subspace_range / 2.0
        inputs = jax.random.uniform(rng, (batch_size, self.n_features), jnp.float32, -bound, bound)
        inputs += jnp.expand_dims(self.input_subspace_centers, 0)
        min_val, max_val = self.input_bounds
        inputs = min_val + jnp.mod(inputs - min_val, max_val - min_val)
        return inputs
    
    def generate_batch(self, batch_size: int = 1) -> Tuple[eqx.Module, Tuple]:
        """Generates a single batch of data.
        
        Args:
            batch_size: Size of batch to generate
            
        Returns:
            Tuple containing:
            - New task state
            - Batch data (x, y)
        """
        accumulators = self.flip_accumulators + self.flip_rate * self._n_flippable * batch_size
        
        new_rng, flip_key, x_key = random.split(self.rng, 3)

        # Flip weights according to accumulators
        new_weights = [self.weights[i] for i in range(self.n_stationary_layers)]
        new_accumulators = [0.0 for _ in range(self.n_stationary_layers)]
        flip_keys = random.split(flip_key, len(self.weights))
        for layer_idx in range(self.n_stationary_layers, len(self.weights)):
            new_weight, accumulator = self._flip_signs(
                weights = self.weights[layer_idx],
                flip_accumulator = accumulators[layer_idx],
                key = flip_keys[layer_idx],
            )
            new_weights.append(new_weight)
            new_accumulators.append(accumulator)
        new_accumulators = jnp.array(new_accumulators, dtype=jnp.float32)
        
        input_subspace_centers = self.input_subspace_centers
        if self.input_change_freq is not None:
            updated_input_subspace_centers = self._compute_updated_input_subspace_centers(x_key)
            update_mask = jnp.ones_like(input_subspace_centers)
            update_mask = update_mask * (self.step % self.input_change_freq == 0)
            input_subspace_centers = jnp.where(update_mask, updated_input_subspace_centers, input_subspace_centers)
        
        # Create a temporary task with the updated weights and input distribution
        new_task_state: InputChangingGEOFFTask = tree_replace(
            self,
            weights = new_weights,
            flip_accumulators = new_accumulators,
            input_subspace_centers = input_subspace_centers,
            step = self.step + 1,
            rng = new_rng,
        )
        
        # Generate random input features
        x = new_task_state._sample_inputs(x_key, batch_size)
        
        # Forward pass through target network
        y = jax.vmap(new_task_state._forward)(x)
        
        # Return updated state and the batch
        return new_task_state, (x, y)


class BinaryRegressionTask(InputChangingGEOFFTask):
    """Input changing version of GEOFF task with configurable depth and activation.
    
    This implements the JAX version of the GEOFF task for use with Equinox.
    Model is structured as an Equinox module with separate static and non-static
    parameters as needed.
    """
    output_thresholds: Float[Array, 'n_outputs'] # Thresholds for each output dimension

    def __init__(
        self,
        n_features: int,
        flip_rate: float = 0.0, # Percentage of weights to flip per step
        n_layers: int = 2,
        n_stationary_layers: int = 0,
        hidden_dim: int = 64,
        n_outputs: int = 1,
        weight_scale: float = 1.0,
        sparsity: float = 0.0,
        input_bounds: Tuple[float, float] = (-1.0, 1.0),
        input_subspace_range: float = 0.1,
        input_change_freq: Optional[int] = None, # Int unless inf
        max_input_center_change: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__(
            n_features, flip_rate, n_layers, n_stationary_layers, hidden_dim,
            n_outputs, weight_scale, 'ltu', sparsity, 'binary', input_bounds,
            input_subspace_range, input_change_freq, max_input_center_change, seed,
        )
        self.rng, output_threshold_key = jax.random.split(self.rng, 2)
        self.output_thresholds = self._compute_output_thresholds(output_threshold_key)
    
    def _compute_output_thresholds(self, rng: jax.random.PRNGKey, n_samples: int = 1000):
        inputs = jax.random.uniform(
            rng, (n_samples, self.n_features), jnp.float32,
            self.input_bounds[0], self.input_bounds[1],
        )
        outputs = jax.vmap(self._forward)(inputs)
        return jnp.mean(outputs, axis=0)
    
    def generate_batch(self, batch_size: int = 1) -> Tuple[eqx.Module, Tuple]:
        """Generates a single batch of data.
        
        Args:
            batch_size: Size of batch to generate
            
        Returns:
            Tuple containing:
            - New task state
            - Batch data (x, y)
        """
        new_task_state, (x, y) = super().generate_batch(batch_size)
        y = jnp.where(y > self.output_thresholds, 1, 0)
        return new_task_state, (x, y)


class CoreTransientBinaryTask(eqx.Module):
    """Binary regression task with separate core and transient layers.
    
    Core layers form the initial representation with standard LTU activation
    (threshold at 0). Transient layers have thresholds calibrated to achieve a
    target activation rate, creating sparse representations.
    
    The network structure is:
    - Input -> Core layers (with core_hidden_dim) -> Transient layers 
      (with transient_hidden_dim) -> Output
    
    Layer biases are calibrated such that transient layer units fire at the
    target activation rate when sampled uniformly from the full input bounds.
    Output thresholds are computed after layer bias calibration to maximize
    output entropy (balanced class distribution).
    """
    
    # Static parameters (configuration)
    n_features: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    weight_scale: float = eqx.field(static=True)
    
    # Core layer configuration
    n_core_layers: int = eqx.field(static=True)
    core_hidden_dim: int = eqx.field(static=True)
    core_sparsity: float = eqx.field(static=True)
    
    # Transient layer configuration
    transient_hidden_dim: int = eqx.field(static=True)
    transient_sparsity: float = eqx.field(static=True)
    transient_activation_rate: float = eqx.field(static=True)
    
    # Input distribution configuration
    input_bounds: Tuple[float, float] = eqx.field(static=True)
    input_subspace_range: float = eqx.field(static=True)
    input_change_freq: Optional[int] = eqx.field(static=True)
    max_input_center_change: float = eqx.field(static=True)
    n_calibration_samples: int = eqx.field(static=True)
    
    # Dynamic parameters (weights and state)
    weights: List[Float[Array, 'in_features out_features']]
    layer_biases: List[Float[Array, 'layer_dim']]
    output_thresholds: Float[Array, 'n_outputs']
    input_subspace_centers: Float[Array, 'n_features']
    step: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_features: int,
        n_core_layers: int = 2,
        core_hidden_dim: int = 64,
        core_sparsity: float = 0.0,
        n_transient_layers: int = 2,
        transient_hidden_dim: int = 2048,
        transient_sparsity: float = 0.8,
        transient_activation_rate: float = 0.05,
        n_outputs: int = 1,
        weight_scale: float = 1.0,
        input_bounds: Tuple[float, float] = (-1.0, 1.0),
        input_subspace_range: float = 0.1,
        input_change_freq: Optional[int] = None,
        max_input_center_change: float = 0.1,
        n_calibration_samples: int = 10000,
        seed: Optional[int] = None,
    ):
        """Initialize a core-transient binary regression task.
        
        Args:
            n_features: Number of input features
            n_core_layers: Number of core layers (with standard threshold at 0)
            core_hidden_dim: Hidden dimension for core layers
            core_sparsity: Weight sparsity for core layers (fraction set to zero)
            n_transient_layers: Number of transient layers (with calibrated thresholds)
            transient_hidden_dim: Hidden dimension for transient layers
            transient_sparsity: Weight sparsity for transient layers
            transient_activation_rate: Target activation rate for transient units
            n_outputs: Number of output dimensions
            weight_scale: Scale factor for weights
            input_bounds: Overall bounds of the input space
            input_subspace_range: Range of the uniform distributions for sampling
            input_change_freq: Number of steps between input changes
            max_input_center_change: Maximum change of a subspace center per step
            n_calibration_samples: Number of samples for calibrating thresholds
            seed: Random seed for reproducibility
        """
        # Store static configuration
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_layers = n_core_layers + n_transient_layers + 1
        self.weight_scale = weight_scale
        
        # Core layer configuration
        self.n_core_layers = n_core_layers
        self.core_hidden_dim = core_hidden_dim
        self.core_sparsity = core_sparsity
        
        # Transient layer configuration
        self.transient_hidden_dim = transient_hidden_dim
        self.transient_sparsity = transient_sparsity
        self.transient_activation_rate = transient_activation_rate
        
        # Input distribution configuration
        self.input_bounds = input_bounds
        self.input_subspace_range = input_subspace_range
        self.input_change_freq = input_change_freq
        self.max_input_center_change = max_input_center_change
        self.n_calibration_samples = n_calibration_samples
        
        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)
        
        # Initialize network weights
        key, network_key = random.split(key)
        self._initialize_network(network_key)
        
        # Initialize input distribution
        key, input_key = random.split(key)
        self.input_subspace_centers = random.uniform(
            input_key, (n_features,), jnp.float32,
            input_bounds[0], input_bounds[1],
        )
        self.step = jnp.array(0, dtype=jnp.int32)
        
        # Calibrate layer biases for transient layers
        key, calibrate_key = random.split(key)
        self._calibrate_layer_biases(calibrate_key)
        
        # Compute output thresholds after calibrating layer biases
        key, threshold_key = random.split(key)
        self.output_thresholds = self._compute_output_thresholds(threshold_key)
        
        self.rng = key
    
    def _initialize_network(self, key: random.PRNGKey):
        """Initialize network weights with different dimensions for core vs transient."""
        keys = random.split(key, 2 * self.n_layers)
        weight_keys = keys[:self.n_layers]
        sparsify_keys = keys[self.n_layers:]
        
        all_weights = []
        n_hidden_layers = self.n_layers - 1
        
        for i in range(self.n_layers):
            # Determine input dimension
            if i == 0:
                in_dim = self.n_features
            elif i - 1 < self.n_core_layers:
                in_dim = self.core_hidden_dim
            else:
                in_dim = self.transient_hidden_dim
            
            # Determine output dimension
            if i == self.n_layers - 1:
                out_dim = self.n_outputs
            elif i < self.n_core_layers:
                out_dim = self.core_hidden_dim
            else:
                out_dim = self.transient_hidden_dim
            
            # Determine sparsity for this layer
            if i == self.n_layers - 1:
                layer_sparsity = 0.0
            elif i < self.n_core_layers:
                layer_sparsity = self.core_sparsity
            else:
                layer_sparsity = self.transient_sparsity
            
            # Initialize weights
            w = initialize_weights(
                weight_keys[i], in_dim, out_dim,
                weight_init = 'binary',
                weight_scale = self.weight_scale,
            )
            if layer_sparsity > 0:
                w = sparsify_weights(sparsify_keys[i], w, layer_sparsity)
            all_weights.append(w)
        
        self.weights = all_weights
        
        # Initialize biases to zeros (will be calibrated later)
        self.layer_biases = []
        for i in range(n_hidden_layers):
            if i < self.n_core_layers:
                bias_dim = self.core_hidden_dim
            else:
                bias_dim = self.transient_hidden_dim
            self.layer_biases.append(jnp.zeros(bias_dim))
    
    def _forward(self, x: jax.Array) -> jax.Array:
        """Forward pass with layer-specific biases (LTU activation)."""
        for i in range(self.n_layers - 1):
            x = x @ self.weights[i]
            x = x - self.layer_biases[i]
            x = jnp.where(x > 0, 1.0, 0.0)  # LTU activation
        
        return x @ self.weights[-1]
    
    def _calibrate_layer_biases(self, key: random.PRNGKey):
        """Calibrate biases for each layer to achieve target activation rates.
        
        Core layers keep bias at 0 (standard LTU threshold).
        Transient layers have biases set to the appropriate percentile of
        pre-activations so that each unit fires at the target activation rate.
        """
        full_input_samples = random.uniform(
            key,
            (self.n_calibration_samples, self.n_features),
            jnp.float32,
            self.input_bounds[0],
            self.input_bounds[1],
        )
        
        calibrated_biases = []
        x = full_input_samples
        n_hidden_layers = self.n_layers - 1
        
        for layer_idx in range(n_hidden_layers):
            pre_activations = x @ self.weights[layer_idx]
            
            if layer_idx >= self.n_core_layers:
                # Transient layer: set threshold for target activation rate
                target_percentile = 100 * (1 - self.transient_activation_rate)
                thresholds = jnp.percentile(pre_activations, target_percentile, axis = 0)
                calibrated_biases.append(thresholds)
            else:
                # Core layer: keep bias at 0
                calibrated_biases.append(self.layer_biases[layer_idx])
            
            # Apply bias and activation for next layer
            x = pre_activations - calibrated_biases[layer_idx]
            x = jnp.where(x > 0, 1.0, 0.0)
        
        self.layer_biases = calibrated_biases
    
    def _compute_output_thresholds(
        self, key: random.PRNGKey,
    ) -> Float[Array, 'n_outputs']:
        """Compute output thresholds to maximize entropy (balance class distribution)."""
        inputs = random.uniform(
            key, (self.n_calibration_samples, self.n_features), jnp.float32,
            self.input_bounds[0], self.input_bounds[1],
        )
        outputs = jax.vmap(self._forward)(inputs)
        return jnp.mean(outputs, axis = 0)
    
    def _compute_updated_input_subspace_centers(
        self, key: random.PRNGKey,
    ) -> Float[Array, 'n_features']:
        """Compute new input subspace centers with random shift."""
        center_shifts = random.uniform(
            key, (self.n_features,), jnp.float32,
            -self.max_input_center_change, self.max_input_center_change,
        )
        new_centers = self.input_subspace_centers + center_shifts
        min_bound, max_bound = self.input_bounds
        range_size = max_bound - min_bound
        new_centers = min_bound + jnp.mod(new_centers - min_bound, range_size)
        return new_centers
    
    def _sample_inputs(
        self, key: random.PRNGKey, batch_size: int = 1,
    ) -> Float[Array, 'batch_size n_features']:
        """Sample inputs from current subspace."""
        bound = self.input_subspace_range / 2.0
        inputs = random.uniform(
            key, (batch_size, self.n_features), jnp.float32, -bound, bound,
        )
        inputs = inputs + jnp.expand_dims(self.input_subspace_centers, 0)
        min_val, max_val = self.input_bounds
        inputs = min_val + jnp.mod(inputs - min_val, max_val - min_val)
        return inputs
    
    def generate_batch(self, batch_size: int = 1) -> Tuple[eqx.Module, Tuple]:
        """Generates a single batch of data.
        
        Args:
            batch_size: Size of batch to generate
            
        Returns:
            Tuple containing:
            - New task state
            - Batch data (x, y) where y is binary
        """
        new_rng, center_key, x_key = random.split(self.rng, 3)
        
        # Update input subspace centers if needed
        input_subspace_centers = self.input_subspace_centers
        if self.input_change_freq is not None:
            updated_centers = self._compute_updated_input_subspace_centers(center_key)
            should_update = (self.step % self.input_change_freq == 0)
            input_subspace_centers = jnp.where(
                should_update, updated_centers, input_subspace_centers,
            )
        
        # Create new task state
        new_task_state: CoreTransientBinaryTask = tree_replace(
            self,
            input_subspace_centers = input_subspace_centers,
            step = self.step + 1,
            rng = new_rng,
        )
        
        # Generate inputs and compute outputs
        x = new_task_state._sample_inputs(x_key, batch_size)
        y = jax.vmap(new_task_state._forward)(x)
        y = jnp.where(y > self.output_thresholds, 1, 0)
        
        return new_task_state, (x, y)
