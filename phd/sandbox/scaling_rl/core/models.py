"""Network architecture for StreamAC experiments.

StreamACNet default architecture matches the reference streaming-drl:
  - Sparse weight initialization (90% sparsity), zero biases
  - LayerNorm + configurable activation after each hidden layer
  - Plain linear output layer (no norm, no activation)
"""
import math
from typing import Callable, List

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray


ACTIVATION_MAP: dict[str, Callable] = {
    'leaky_relu': jax.nn.leaky_relu,
    'relu': jax.nn.relu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
}


def _sparse_init_layer(layer: nn.Linear, sparsity: float, key: PRNGKeyArray) -> nn.Linear:
    """Apply sparse init to a Linear layer: zero-out `sparsity` fraction of each row.

    Uses vmap over output neurons instead of a Python loop — fast even for large
    hidden_dim.  n_zeros is capped at in_features-1 so at least one incoming weight
    remains active (prevents dead neurons when in_features is small, e.g. CartPole).
    """
    w = layer.weight                    # (out_features, in_features)
    out_features, in_features = w.shape
    n_zeros = min(int(math.ceil(sparsity * in_features)), in_features - 1)

    row_keys = jax.random.split(key, out_features)

    def _mask_row(k: PRNGKeyArray) -> Array:
        perm = jax.random.permutation(k, in_features)
        # Set the first n_zeros positions in the permutation to zero
        return jnp.ones(in_features, dtype=w.dtype).at[perm[:n_zeros]].set(0.0)

    masks = jax.vmap(_mask_row)(row_keys)   # (out_features, in_features)
    new_w = w * masks

    new_layer = eqx.tree_at(lambda l: l.weight, layer, new_w)
    if layer.bias is not None:
        new_layer = eqx.tree_at(lambda l: l.bias, new_layer, jnp.zeros_like(layer.bias))
    return new_layer


class StreamACNet(eqx.Module):
    """Actor or critic network for StreamAC.

    Default architecture per hidden layer: Linear → LayerNorm → LeakyReLU
    Output layer: Linear (no norm, no activation)
    """
    linear_layers: List[nn.Linear]
    layer_norms: List[nn.LayerNorm]
    activation_fn: Callable = eqx.field(static=True)
    n_hidden: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int,
        hidden_dim: int,
        activation: str = 'leaky_relu',
        weight_init_method: str = 'sparse',
        sparsity: float = 0.9,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            input_dim:          Observation dimension.
            output_dim:         Number of outputs (actions for actor, 1 for critic).
            n_layers:           Total layers including output (n_layers=2 → 1 hidden + 1 output).
            hidden_dim:         Width of hidden layers.
            activation:         Hidden-layer activation ('leaky_relu', 'relu', 'tanh', 'sigmoid').
            weight_init_method: 'sparse' (default, 0.9 sparsity + zero biases) or 'lecun_uniform'.
            sparsity:           Sparsity fraction when weight_init_method='sparse'.
            key:                PRNG key.
        """
        assert n_layers >= 2, "n_layers must be >= 2 (at least 1 hidden + output)"
        assert activation in ACTIVATION_MAP, f"activation must be one of {list(ACTIVATION_MAP)}"
        assert weight_init_method in ('sparse', 'lecun_uniform'), \
            "weight_init_method must be 'sparse' or 'lecun_uniform'"

        n_hidden = n_layers - 1
        keys = jax.random.split(key, n_layers)

        linear_layers: List[nn.Linear] = []
        layer_norms: List[nn.LayerNorm] = []

        # Hidden layers
        prev_dim = input_dim
        for i in range(n_hidden):
            layer = nn.Linear(prev_dim, hidden_dim, use_bias=True, key=keys[i])
            if weight_init_method == 'sparse':
                layer = _sparse_init_layer(layer, sparsity, keys[i])
            linear_layers.append(layer)
            layer_norms.append(nn.LayerNorm((hidden_dim,)))
            prev_dim = hidden_dim

        # Output layer
        out_layer = nn.Linear(hidden_dim, output_dim, use_bias=True, key=keys[-1])
        if weight_init_method == 'sparse':
            out_layer = _sparse_init_layer(out_layer, sparsity, keys[-1])
        linear_layers.append(out_layer)

        self.linear_layers = linear_layers
        self.layer_norms = layer_norms
        self.activation_fn = ACTIVATION_MAP[activation]
        self.n_hidden = n_hidden

    def __call__(self, x: Array) -> Array:
        for layer, ln in zip(self.linear_layers[:self.n_hidden], self.layer_norms):
            x = self.activation_fn(ln(layer(x)))
        return self.linear_layers[-1](x)
