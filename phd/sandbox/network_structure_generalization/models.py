import math
from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import numpy as np


class MLP(eqx.Module):
    """Simple feedforward network for binary classification."""
    layers: list
    activation: str = eqx.field(static=True)

    def __init__(self, layer_dims: List[int], activation: str = 'relu', *, key):
        keys = random.split(key, len(layer_dims) - 1)
        self.layers = [
            eqx.nn.Linear(layer_dims[i], layer_dims[i + 1], key=keys[i])
            for i in range(len(layer_dims) - 1)
        ]
        self.activation = activation

    def __call__(self, x):
        x = x.flatten()
        act_fn = getattr(jax.nn, self.activation)
        for layer in self.layers[:-1]:
            x = act_fn(layer(x))
        return self.layers[-1](x)


class LocallyConnectedLayer(eqx.Module):
    """Convolution layer without weight sharing (separate weights per spatial position).

    Uses jax.lax.conv_general_dilated_local with LeCun uniform initialization.
    """
    weight: jax.Array   # (in_c*kH*kW, out_c, out_H, out_W)
    bias: Optional[jax.Array]  # (out_c, out_H, out_W) or None
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    padding: str = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, spatial_shape: tuple,
                 kernel_size: int = 3, stride: int = 1, padding: str = 'SAME',
                 use_bias: bool = True, *, key):
        h, w = spatial_shape
        if padding == 'SAME':
            out_h = (h + stride - 1) // stride
            out_w = (w + stride - 1) // stride
        else:  # VALID
            out_h = (h - kernel_size + stride) // stride
            out_w = (w - kernel_size + stride) // stride

        fan_in = in_channels * kernel_size * kernel_size
        bound = math.sqrt(3.0 / fan_in)

        k1, k2 = random.split(key)
        self.weight = random.uniform(
            k1, (fan_in, out_channels, out_h, out_w),
            minval=-bound, maxval=bound,
        )
        self.bias = jnp.zeros((out_channels, out_h, out_w)) if use_bias else None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        """Forward pass. x: (C_in, H, W) -> (C_out, H_out, W_out)"""
        x = x[None]  # (1, C_in, H, W)
        y = jax.lax.conv_general_dilated_local(
            x, self.weight,
            window_strides=(self.stride, self.stride),
            padding=self.padding,
            filter_shape=(self.kernel_size, self.kernel_size),
            dimension_numbers=('NCHW', 'IOHW', 'NCHW'),
        )
        y = y[0]  # (C_out, H_out, W_out)
        if self.bias is not None:
            y = y + self.bias
        return y


class LocallyConnectedCNN(eqx.Module):
    """CNN using locally connected layers (no weight sharing)."""
    conv_layers: list
    fc_layer: eqx.nn.Linear
    activation: str = eqx.field(static=True)

    def __init__(self, image_shape: tuple, channels: List[int],
                 kernel_size: int = 3, stride: int = 2,
                 activation: str = 'relu', *, key):
        in_c, h, w = image_shape
        keys = random.split(key, len(channels) + 1)

        self.conv_layers = []
        current_c = in_c
        current_h, current_w = h, w
        for i, out_c in enumerate(channels):
            layer = LocallyConnectedLayer(
                current_c, out_c, (current_h, current_w),
                kernel_size=kernel_size, stride=stride,
                padding='SAME', key=keys[i],
            )
            current_c = out_c
            current_h = (current_h + stride - 1) // stride
            current_w = (current_w + stride - 1) // stride
            self.conv_layers.append(layer)

        fc_in = current_c * current_h * current_w
        self.fc_layer = eqx.nn.Linear(fc_in, 2, key=keys[-1])
        self.activation = activation

    def __call__(self, x):
        act_fn = getattr(jax.nn, self.activation)
        for layer in self.conv_layers:
            x = act_fn(layer(x))
        x = x.flatten()
        return self.fc_layer(x)


# Default architectures targeting ~50k parameters
# MNIST MLP: 784*63 + 63 + 63*2 + 2 = 49,583
# CIFAR-100 MLP: 3072*16 + 16 + 16*2 + 2 = 49,202
# MNIST CNN: 2 locally connected layers [8, 9] channels ≈ 48,757
# CIFAR-100 CNN: 2 locally connected layers [5, 5] channels ≈ 51,202
DEFAULT_CONFIGS = {
    ('mlp', 'mnist'): {'layer_dims': [784, 63, 2]},
    ('mlp', 'cifar100'): {'layer_dims': [3072, 16, 2]},
    ('cnn', 'mnist'): {'channels': [8, 9]},
    ('cnn', 'cifar100'): {'channels': [5, 5]},
}


def build_model(args, image_shape, *, key):
    """Factory function to build a model based on args."""
    if args.model == 'mlp':
        if args.mlp_hidden_dims:
            input_dim = int(np.prod(image_shape))
            layer_dims = [input_dim] + args.mlp_hidden_dims + [2]
        else:
            layer_dims = DEFAULT_CONFIGS[('mlp', args.dataset)]['layer_dims']
        return MLP(layer_dims, activation=args.activation, key=key)
    elif args.model == 'cnn':
        channels = args.cnn_channels or DEFAULT_CONFIGS[('cnn', args.dataset)]['channels']
        return LocallyConnectedCNN(
            image_shape, channels,
            kernel_size=args.cnn_kernel_size,
            stride=args.cnn_stride,
            activation=args.activation,
            key=key,
        )
    else:
        raise ValueError(f'Unknown model: {args.model}')


