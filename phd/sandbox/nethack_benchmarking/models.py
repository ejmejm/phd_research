from typing import List, Optional, Tuple, Callable

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


# Activation function mapping
ACTIVATION_MAP = {
    'relu': jax.nn.relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
    'swish': jax.nn.swish,
    'linear': jax.nn.identity,
}


def _compute_output_size(input_size: int, kernel_size: int, stride: int, padding: str = 'SAME') -> int:
    """Compute output size after a convolution layer.
    
    Args:
        input_size: Input spatial dimension (height or width)
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding mode ('SAME' or 'VALID')
        
    Returns:
        Output spatial dimension
    """
    if padding == 'SAME':
        # SAME padding preserves input size when stride=1, otherwise it's ceil(input_size / stride)
        return (input_size + stride - 1) // stride
    else:  # VALID padding
        return (input_size - kernel_size) // stride + 1


class ConvNet(eqx.Module):
    """Convolutional neural network with configurable architecture.
    
    The network consists of:
    1. An embedding layer that maps input channels to a specified embedding dimension
    2. A series of configurable convolutional layers
    3. A series of configurable fully connected layers
    
    Input shape: (n_channels, height, width)
    """
    
    input_shape: Tuple[int, int, int] = eqx.field(static=True)  # (n_channels, height, width)
    embedding_dim: int = eqx.field(static=True)
    conv_configs: List[Tuple[int, int, int]] = eqx.field(static=True)  # (kernel_size, stride, out_channels)
    fc_hidden_dims: List[int] = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    activation_fn: Callable = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    
    embedding: nn.Linear
    conv_layers: List[nn.Conv2d]
    fc_layers: List[nn.Linear]
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (n_channels, height, width)
        embedding_dim: int,
        conv_configs: List[Tuple[int, int, int]],  # List of (kernel_size, stride, out_channels)
        fc_hidden_dims: List[int],  # List of hidden dimensions for FC layers
        output_dim: int,
        activation: str = 'relu',
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the ConvNet.
        
        Args:
            input_shape: Tuple of (n_channels, height, width) for input shape
            embedding_dim: Dimension of the embedding layer output
            conv_configs: List of tuples (kernel_size, stride, out_channels) for each conv layer
            fc_hidden_dims: List of hidden dimensions for each fully connected layer
            output_dim: Dimension of the final output
            activation: Activation function to use ('relu', 'tanh', 'sigmoid', 'swish', 'linear')
            use_bias: Whether to use bias in layers
            key: PRNG key for weight initialization
        """
        self.input_shape = input_shape
        input_channels, input_height, input_width = input_shape
        self.embedding_dim = embedding_dim
        self.conv_configs = conv_configs
        self.fc_hidden_dims = fc_hidden_dims
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # Get activation function
        assert activation in ACTIVATION_MAP, f"Invalid activation: {activation}. Must be one of {list(ACTIVATION_MAP.keys())}"
        self.activation_fn = ACTIVATION_MAP[activation]
        
        # Split keys for layer initialization
        n_conv_layers = len(conv_configs)
        n_fc_layers = len(fc_hidden_dims) + 1  # hidden + output
        keys = jax.random.split(key, 1 + n_conv_layers + n_fc_layers)  # embedding + convs + fcs
        key_idx = 0
        
        # Create embedding layer
        # Embed the channels dimension: (input_channels, H, W) -> (embedding_dim, H, W)
        self.embedding = nn.Linear(
            input_channels,
            embedding_dim,
            use_bias=use_bias,
            key=keys[key_idx],
        )
        key_idx += 1
        
        # Create convolutional layers
        self.conv_layers = []
        in_channels = embedding_dim
        current_height = input_height
        current_width = input_width
        
        for kernel_size, stride, out_channels in conv_configs:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='SAME',  # Preserve spatial dimensions (approximately)
                    use_bias=use_bias,
                    key=keys[key_idx],
                )
            )
            # Compute output spatial dimensions
            current_height = _compute_output_size(current_height, kernel_size, stride, padding='SAME')
            current_width = _compute_output_size(current_width, kernel_size, stride, padding='SAME')
            in_channels = out_channels
            key_idx += 1
        
        # Compute flattened size after conv layers
        fc_input_dim = in_channels * current_height * current_width
        
        # Create fully connected layers
        self.fc_layers = []
        
        if len(fc_hidden_dims) > 0:
            # First hidden layer
            self.fc_layers.append(
                nn.Linear(
                    fc_input_dim,
                    fc_hidden_dims[0],
                    use_bias=use_bias,
                    key=keys[key_idx],
                )
            )
            key_idx += 1
            
            # Remaining hidden layers
            for i in range(1, len(fc_hidden_dims)):
                self.fc_layers.append(
                    nn.Linear(
                        fc_hidden_dims[i - 1],
                        fc_hidden_dims[i],
                        use_bias=use_bias,
                        key=keys[key_idx],
                    )
                )
                key_idx += 1
            
            # Output layer
            self.fc_layers.append(
                nn.Linear(
                    fc_hidden_dims[-1],
                    output_dim,
                    use_bias=use_bias,
                    key=keys[-1],
                )
            )
        else:
            # No hidden FC layers, just output layer
            self.fc_layers.append(
                nn.Linear(
                    fc_input_dim,
                    output_dim,
                    use_bias=use_bias,
                    key=keys[-1],
                )
            )
    
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """Forward pass through the network.
        
        Args:
            x: Input array of shape (n_channels, height, width)
            key: Optional PRNG key (not used but kept for interface compatibility)
            
        Returns:
            Output array of shape (output_dim,)
        """
        # Input shape: (n_channels, height, width)
        n_channels, height, width = x.shape
        
        # Apply embedding layer
        # Reshape to (height, width, n_channels) for easier processing
        # Then apply linear layer to channels dimension
        x = jnp.moveaxis(x, 0, -1)  # (height, width, n_channels)
        x = x.reshape(-1, n_channels)  # (height * width, n_channels)
        x = self.embedding(x)  # (height * width, embedding_dim)
        x = x.reshape(height, width, self.embedding_dim)  # (height, width, embedding_dim)
        x = jnp.moveaxis(x, -1, 0)  # (embedding_dim, height, width)
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)  # (out_channels, height', width')
            x = self.activation_fn(x)
        
        # Flatten for fully connected layers
        x = x.flatten()  # (out_channels * height' * width',)
        
        # Apply fully connected layers
        for fc_layer in self.fc_layers[:-1]:
            x = fc_layer(x)
            x = self.activation_fn(x)
        
        # Output layer (no activation)
        x = self.fc_layers[-1](x)
        
        return x
