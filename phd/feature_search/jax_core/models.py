from functools import partial
from typing import Callable, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from jax import Array
from jaxtyping import PRNGKeyArray


@partial(jax.jit, static_argnums=(1, 2))
def lecun_uniform(key: PRNGKeyArray, shape: Tuple[int, ...], in_dim: Optional[int] = None) -> Array:
    """LeCun uniform initialization."""
    in_dim = shape[-1] if in_dim is None else in_dim
    bound = jnp.sqrt(3.0) / jnp.sqrt(in_dim)
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)


@partial(jax.jit, static_argnames=('shape', 'in_dim',))
def kaiming_uniform(key: PRNGKeyArray, shape: Tuple[int, ...], gain: float = 1.0, in_dim: Optional[int] = None) -> Array:
    """Kaiming uniform initialization."""
    in_dim = shape[-1] if in_dim is None else in_dim
    bound = gain * jnp.sqrt(3.0) / jnp.sqrt(in_dim)
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)


def ltu(x: Array, threshold: float = 0.0) -> Array:
    """Linear threshold unit with sigmoid-like gradient.
    
    Forward pass applies step function but backward pass uses sigmoid gradient.
    
    Args:
        x: Input array
        threshold: Threshold value for step function (default 0.0)
        
    Returns:
        Array with 1s where x > threshold and 0s elsewhere
    """
    @jax.custom_vjp
    def ltu_fn(x):
        return (x > threshold).astype(jnp.float32)
    
    def ltu_fwd(x):
        return ltu_fn(x), x
    
    def ltu_bwd(res, g):
        x = res
        # Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_val = jax.nn.sigmoid(x - threshold) 
        grad = sigmoid_val * (1 - sigmoid_val)
        return (g * grad,)
    
    ltu_fn.defvjp(ltu_fwd, ltu_bwd)
    return ltu_fn(x)


# Activation function mapping
ACTIVATION_MAP = {
    'relu': jax.nn.relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
    'ltu': ltu,
}


class MLP(eqx.Module):
    """Multi-layer perceptron with configurable architecture and initialization."""
    
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    weight_init_method: str = eqx.field(static=True)
    n_frozen_layers: int = eqx.field(static=True)
    activation_fn: Callable = eqx.field(static=True)
    
    layers: List[Any]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int,
        hidden_dim: int,
        weight_init_method: str,
        activation: str = 'tanh',
        n_frozen_layers: int = 0,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            n_layers: Number of layers (including output)
            hidden_dim: Size of hidden layers
            weight_init_method: How to initialize weights ('zeros', 'kaiming', or 'binary')
            activation: Activation function ('relu', 'tanh', 'sigmoid', or 'ltu')
            n_frozen_layers: Number of frozen layers (note: freezing is handled differently in JAX)
            key: PRNG key for weight initialization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.weight_init_method = weight_init_method
        self.n_frozen_layers = n_frozen_layers
        
        # Get activation function
        self.activation_fn = ACTIVATION_MAP[activation]
        
        # Split keys for layer initialization
        if n_layers == 1:
            keys = [key]
        else:
            keys = jax.random.split(key, n_layers)
        
        # Build layers
        self.layers = []
        if n_layers == 1:
            layer = self._create_linear_layer(keys[0], input_dim, output_dim, weight_init_method)
            self.layers.append(layer)
        else:
            # First layer
            layer = self._create_linear_layer(keys[0], input_dim, hidden_dim, weight_init_method)
            self.layers.append(layer)
            
            # Hidden layers
            for i in range(1, n_layers - 1):
                layer = self._create_linear_layer(keys[i], hidden_dim, hidden_dim)
                self.layers.append(layer)
            
            # Output layer
            layer = self._create_linear_layer(keys[-1], hidden_dim, output_dim)
            self.layers.append(layer)
    
    def _create_linear_layer(
            self,
            key: PRNGKeyArray,
            in_features: int,
            out_features: int,
            init_method: str = 'kaiming_uniform',
    ) -> nn.Linear:
        """Create a linear layer with specified initialization."""
        if init_method == 'zeros':
            # Create layer and then zero out the weights
            layer = nn.Linear(in_features, out_features, use_bias=False, key=key)
            layer = eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
            return layer
        elif init_method == 'kaiming_uniform':
            # Use default Equinox initialization (which is similar to Kaiming)
            return nn.Linear(in_features, out_features, use_bias=False, key=key)
        elif init_method == 'binary':
            # Create layer with binary weights (-1 or 1)
            layer = nn.Linear(in_features, out_features, use_bias=False, key=key)
            binary_weights = jax.random.randint(key, layer.weight.shape, 0, 2).astype(jnp.float32) * 2 - 1
            layer = eqx.tree_at(lambda l: l.weight, layer, binary_weights)
            return layer
        else:
            raise ValueError(f'Invalid weight initialization method: {init_method}')
    
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Tuple[Array, List[Array]]:
        """Forward pass through the network.
        
        Args:
            x: Input array
            key: Optional PRNG key (not used but kept for interface compatibility)
            
        Returns:
            Tuple of (output, param_inputs) where param_inputs is a list of inputs to each layer
            (in order: input to layer 0, input to layer 1, ..., input to final layer)
        """
        param_inputs = []
        
        # Process all layers except the last
        for i, layer in enumerate(self.layers[:-1]):
            param_inputs.append(x)
            x = layer(x)
            x = self.activation_fn(x)
        
        # Last layer (output layer)
        param_inputs.append(x)
        output = self.layers[-1](x)
        
        # # Structure param_inputs to match the structure of the model
        # param_inputs = jax.tree.unflatten(jax.tree.structure(self), param_inputs)
        
        return output, param_inputs
    
    def get_first_layer_weights(self) -> Array:
        """Returns the weights of the first layer for utility calculation."""
        return self.layers[0].weight
