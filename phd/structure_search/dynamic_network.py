from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray

from phd.jax_core.models import lecun_uniform, ACTIVATION_MAP


class DynamicNetwork(eqx.Module):
    """Dynamic neural network with preallocated max-capacity arrays.

    Supports variable structure (adding/removing hidden units and connections)
    without JAX recompilation by using constant-sized arrays with padding and
    masking.

    Weight storage uses (units, connections) ordering so that axis=-1 sums
    over connections per unit — required for IDBD autostep compatibility.
    Non-trainable fields use int32 so eqx.filter_value_and_grad (which defaults
    to eqx.is_inexact_array) automatically excludes them from differentiation.
    """

    # Static config (won't cause recompilation when unchanged)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    max_layers: int = eqx.field(static=True)
    max_units_per_layer: int = eqx.field(static=True)
    max_connections_per_unit: int = eqx.field(static=True)
    buffer_size: int = eqx.field(static=True)
    activation_fns: tuple = eqx.field(static=True)

    # Trainable parameters (float32 — included in gradients)
    weights: Array           # (max_layers, max_units_per_layer, max_connections_per_unit)
    output_weights: Array    # (output_dim, buffer_size)

    # Structure arrays (int32 — excluded from gradients)
    input_indices: Array     # (max_layers, max_units_per_layer, max_connections_per_unit)
    unit_mask: Array         # (max_layers, max_units_per_layer)
    activation_indices: Array  # (max_layers, max_units_per_layer)
    output_mask: Array       # (output_dim, buffer_size)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_layers: int,
        max_units_per_layer: int,
        max_connections_per_unit: int,
        activations: Tuple[str, ...] = ('relu',),
        *,
        key: PRNGKeyArray,
    ):
        """Initialize as a linear model (no hidden units, inputs → outputs).

        Args:
            input_dim: Input dimensionality (e.g. 3072 for flattened CIFAR-10).
            output_dim: Number of outputs (e.g. 10 for CIFAR-10).
            max_layers: Maximum number of hidden layers.
            max_units_per_layer: Maximum hidden units per layer.
            max_connections_per_unit: Maximum incoming connections per hidden unit.
            activations: Tuple of activation function names from ACTIVATION_MAP.
            key: PRNG key for weight initialization.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layers = max_layers
        self.max_units_per_layer = max_units_per_layer
        self.max_connections_per_unit = max_connections_per_unit
        self.buffer_size = input_dim + max_layers * max_units_per_layer

        # Resolve activation functions
        self.activation_fns = tuple(ACTIVATION_MAP[name] for name in activations)

        # Hidden layer weights — all zeros (no hidden units yet)
        self.weights = jnp.zeros(
            (max_layers, max_units_per_layer, max_connections_per_unit),
            dtype=jnp.float32,
        )

        # Output layer — lecun_uniform for input-to-output, zeros for rest
        output_w = jnp.zeros((output_dim, self.buffer_size), dtype=jnp.float32)
        output_w = output_w.at[:, :input_dim].set(
            lecun_uniform(key, (output_dim, input_dim), in_dim=input_dim)
        )
        self.output_weights = output_w

        # Input indices — all -1 (no connections)
        self.input_indices = jnp.full(
            (max_layers, max_units_per_layer, max_connections_per_unit),
            fill_value=-1,
            dtype=jnp.int32,
        )

        # Unit mask — all inactive
        self.unit_mask = jnp.zeros(
            (max_layers, max_units_per_layer), dtype=jnp.int32,
        )

        # Activation indices — default to first activation
        self.activation_indices = jnp.zeros(
            (max_layers, max_units_per_layer), dtype=jnp.int32,
        )

        # Output mask — active for input-to-output connections only
        output_m = jnp.zeros((output_dim, self.buffer_size), dtype=jnp.int32)
        output_m = output_m.at[:, :input_dim].set(1)
        self.output_mask = output_m

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Forward pass through the dynamic network.

        Args:
            x: Input array of shape (input_dim,).

        Returns:
            (output, buffer) where output has shape (output_dim,) and buffer
            has shape (buffer_size,) containing all activation values (useful
            for the structure tracker to compute connection utilities).
        """
        buffer = jnp.zeros(self.buffer_size)
        buffer = buffer.at[:self.input_dim].set(x)

        offsets = (
            jnp.arange(self.max_layers) * self.max_units_per_layer + self.input_dim
        )

        def layer_step(buffer, layer_data):
            weights, indices, u_mask, act_idx, offset = layer_data
            # indices shape: (max_units_per_layer, max_connections_per_unit)

            safe_idx = jnp.maximum(indices, 0)
            conn_mask = (indices >= 0).astype(jnp.float32)

            gathered = jnp.take(buffer, safe_idx, mode='clip')
            pre_act = (gathered * conn_mask * weights).sum(axis=-1)

            # Per-unit activation selection via jax.lax.switch
            def apply_activation(idx, val):
                return jax.lax.switch(idx, self.activation_fns, val)

            post_act = jax.vmap(apply_activation)(act_idx, pre_act)
            post_act = post_act * u_mask.astype(jnp.float32)

            buffer = jax.lax.dynamic_update_slice(buffer, post_act, (offset,))
            return buffer, None

        scan_data = (
            self.weights,
            self.input_indices,
            self.unit_mask,
            self.activation_indices,
            offsets,
        )
        buffer, _ = jax.lax.scan(
            layer_step, buffer, scan_data, unroll=min(self.max_layers, 5),
        )

        output = (self.output_weights * self.output_mask.astype(jnp.float32)) @ buffer
        return output, buffer


def count_active_connections(network: DynamicNetwork) -> int:
    """Count the total number of active connections in the network.

    Includes both hidden-layer connections (input_indices >= 0) and
    output-layer connections (output_mask == 1).
    """
    hidden = int(jnp.sum(network.input_indices >= 0))
    output = int(jnp.sum(network.output_mask))
    return hidden + output


def count_active_units(network: DynamicNetwork) -> int:
    """Count the total number of active hidden units."""
    return int(jnp.sum(network.unit_mask))
