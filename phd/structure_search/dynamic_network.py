from typing import Tuple
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray

from phd.jax_core.models import lecun_uniform, ACTIVATION_MAP


# ---------------------------------------------------------------------------
# Custom VJP forward/backward for the hidden-layer scan
# ---------------------------------------------------------------------------

def _forward_impl(
    weights, output_weights, x,
    input_indices, unit_mask, activation_indices, output_mask,
    outgoing_unit_indices, outgoing_conn_indices,
    activation_fns, activation_deriv_fns,
    input_dim, max_layers, max_units_per_layer, buffer_size,
):
    """Shared forward logic (used by both primal and fwd rule)."""
    buffer = jnp.zeros(buffer_size)
    buffer = buffer.at[:input_dim].set(x)

    offsets = jnp.arange(max_layers) * max_units_per_layer + input_dim

    def layer_step(buffer, layer_data):
        w, idx, u_mask, act_idx, offset = layer_data
        safe_idx = jnp.maximum(idx, 0)
        conn_mask = (idx >= 0).astype(jnp.float32)
        gathered = jnp.take(buffer, safe_idx, mode='clip')
        pre_act = (gathered * conn_mask * w).sum(axis=-1)

        def apply_activation(i, val):
            return jax.lax.switch(i, activation_fns, val)

        post_act = jax.vmap(apply_activation)(act_idx, pre_act)
        post_act = post_act * u_mask.astype(jnp.float32)
        buffer = jax.lax.dynamic_update_slice(buffer, post_act, (offset,))
        return buffer, pre_act

    scan_data = (weights, input_indices, unit_mask, activation_indices, offsets)
    buffer, pre_acts = jax.lax.scan(
        layer_step, buffer, scan_data, unroll=min(max_layers, 5),
    )

    output = (output_weights * output_mask.astype(jnp.float32)) @ buffer
    return output, buffer, pre_acts


def _make_dynamic_forward(activation_fns, activation_deriv_fns,
                          input_dim, max_layers, max_units_per_layer, buffer_size):
    """Create a custom_vjp forward function with static config bound.

    Static Python values (functions, ints) are captured in the closure.
    Only JAX arrays are passed as arguments to the custom_vjp function.
    """

    @jax.custom_vjp
    def dynamic_forward(weights, output_weights, x,
                        input_indices, unit_mask, activation_indices, output_mask,
                        outgoing_unit_indices, outgoing_conn_indices):
        output, buffer, _pre_acts = _forward_impl(
            weights, output_weights, x,
            input_indices, unit_mask, activation_indices, output_mask,
            outgoing_unit_indices, outgoing_conn_indices,
            activation_fns, activation_deriv_fns,
            input_dim, max_layers, max_units_per_layer, buffer_size,
        )
        return output, buffer

    def dynamic_forward_fwd(weights, output_weights, x,
                            input_indices, unit_mask, activation_indices, output_mask,
                            outgoing_unit_indices, outgoing_conn_indices):
        output, buffer, pre_acts = _forward_impl(
            weights, output_weights, x,
            input_indices, unit_mask, activation_indices, output_mask,
            outgoing_unit_indices, outgoing_conn_indices,
            activation_fns, activation_deriv_fns,
            input_dim, max_layers, max_units_per_layer, buffer_size,
        )
        residuals = (buffer, pre_acts, weights, output_weights,
                     input_indices, unit_mask, activation_indices, output_mask,
                     outgoing_unit_indices, outgoing_conn_indices)
        return (output, buffer), residuals

    def dynamic_forward_bwd(residuals, g):
        (buffer, pre_acts, weights, output_weights,
         input_indices, unit_mask, activation_indices, output_mask,
         outgoing_unit_indices, outgoing_conn_indices) = residuals
        grad_output, grad_buffer = g

        # --- Output layer ---
        masked_ow = output_weights * output_mask.astype(jnp.float32)
        grad_buffer = masked_ow.T @ grad_output + grad_buffer
        grad_output_weights = jnp.outer(grad_output, buffer) * output_mask.astype(jnp.float32)

        # --- Hidden layers (reverse order, unrolled) ---
        grad_weights = jnp.zeros_like(weights)

        for l in range(max_layers - 1, -1, -1):
            offset = input_dim + l * max_units_per_layer

            # 1. Gradient through post-activation masking
            grad_post_act = jax.lax.dynamic_slice(
                grad_buffer, (offset,), (max_units_per_layer,)
            )
            grad_post_act = grad_post_act * unit_mask[l].astype(jnp.float32)

            # 2. Activation derivative via jax.grad, dispatched with vmap(lax.switch)
            if len(activation_deriv_fns) == 1:
                act_deriv = jax.vmap(activation_deriv_fns[0])(pre_acts[l])
            else:
                def apply_act_deriv(idx, val):
                    return jax.lax.switch(idx, activation_deriv_fns, val)
                act_deriv = jax.vmap(apply_act_deriv)(activation_indices[l], pre_acts[l])
            grad_pre_act = grad_post_act * act_deriv

            # 3. Weight gradients (gather-based, already efficient)
            safe_idx = jnp.maximum(input_indices[l], 0)
            conn_mask = (input_indices[l] >= 0).astype(jnp.float32)
            gathered = jnp.take(buffer, safe_idx, mode='clip')
            grad_weights = grad_weights.at[l].set(
                grad_pre_act[:, None] * gathered * conn_mask
            )

            # 4. Buffer gradient via OUTGOING indices (GATHER, not scatter!)
            out_u = outgoing_unit_indices[l]  # (buffer_size, max_fan_out)
            out_c = outgoing_conn_indices[l]  # (buffer_size, max_fan_out)
            out_mask = (out_u >= 0).astype(jnp.float32)
            safe_u = jnp.maximum(out_u, 0)
            safe_c = jnp.maximum(out_c, 0)

            contrib = grad_pre_act[safe_u] * weights[l, safe_u, safe_c] * out_mask
            grad_buffer = grad_buffer + contrib.sum(axis=-1)

        # 5. Input gradient
        grad_x = grad_buffer[:input_dim]

        # Return cotangents for all 9 arguments (zeros for int32 structure arrays)
        return (grad_weights, grad_output_weights, grad_x,
                jnp.zeros_like(input_indices, dtype=jnp.float32),
                jnp.zeros_like(unit_mask, dtype=jnp.float32),
                jnp.zeros_like(activation_indices, dtype=jnp.float32),
                jnp.zeros_like(output_mask, dtype=jnp.float32),
                jnp.zeros_like(outgoing_unit_indices, dtype=jnp.float32),
                jnp.zeros_like(outgoing_conn_indices, dtype=jnp.float32))

    dynamic_forward.defvjp(dynamic_forward_fwd, dynamic_forward_bwd)
    return dynamic_forward


def _dynamic_forward_plain(
    weights, output_weights, x,
    input_indices, unit_mask, activation_indices, output_mask,
    activation_fns,
    input_dim, max_layers, max_units_per_layer, buffer_size,
):
    """Forward pass WITHOUT custom VJP (for testing gradient correctness)."""
    buffer = jnp.zeros(buffer_size)
    buffer = buffer.at[:input_dim].set(x)

    offsets = jnp.arange(max_layers) * max_units_per_layer + input_dim

    def layer_step(buffer, layer_data):
        w, idx, u_mask, act_idx, offset = layer_data
        safe_idx = jnp.maximum(idx, 0)
        conn_mask = (idx >= 0).astype(jnp.float32)
        gathered = jnp.take(buffer, safe_idx, mode='clip')
        pre_act = (gathered * conn_mask * w).sum(axis=-1)

        def apply_activation(i, val):
            return jax.lax.switch(i, activation_fns, val)

        post_act = jax.vmap(apply_activation)(act_idx, pre_act)
        post_act = post_act * u_mask.astype(jnp.float32)
        buffer = jax.lax.dynamic_update_slice(buffer, post_act, (offset,))
        return buffer, None

    scan_data = (weights, input_indices, unit_mask, activation_indices, offsets)
    buffer, _ = jax.lax.scan(
        layer_step, buffer, scan_data, unroll=min(max_layers, 5),
    )

    output = (output_weights * output_mask.astype(jnp.float32)) @ buffer
    return output, buffer


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
    max_fan_out: int = eqx.field(static=True)
    activation_fns: tuple = eqx.field(static=True)
    activation_deriv_fns: tuple = eqx.field(static=True)

    # Trainable parameters (float32 — included in gradients)
    weights: Array           # (max_layers, max_units_per_layer, max_connections_per_unit)
    output_weights: Array    # (output_dim, buffer_size)

    # Structure arrays (int32 — excluded from gradients)
    input_indices: Array     # (max_layers, max_units_per_layer, max_connections_per_unit)
    unit_mask: Array         # (max_layers, max_units_per_layer)
    activation_indices: Array  # (max_layers, max_units_per_layer)
    output_mask: Array       # (output_dim, buffer_size)
    outgoing_unit_indices: Array  # (max_layers, buffer_size, max_fan_out)
    outgoing_conn_indices: Array  # (max_layers, buffer_size, max_fan_out)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_layers: int,
        max_units_per_layer: int,
        max_connections_per_unit: int,
        activations: Tuple[str, ...] = ('relu',),
        max_fan_out: int | None = None,
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
            max_fan_out: Maximum outgoing connections per buffer position per
                layer. Defaults to max_units_per_layer (fully-connected worst case).
            key: PRNG key for weight initialization.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layers = max_layers
        self.max_units_per_layer = max_units_per_layer
        self.max_connections_per_unit = max_connections_per_unit
        self.buffer_size = input_dim + max_layers * max_units_per_layer
        self.max_fan_out = max_fan_out if max_fan_out is not None else max_units_per_layer

        # Resolve activation functions and their derivatives
        self.activation_fns = tuple(ACTIVATION_MAP[name] for name in activations)
        self.activation_deriv_fns = tuple(jax.grad(fn) for fn in self.activation_fns)

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

        # Outgoing connection indices — all -1 (no connections)
        self.outgoing_unit_indices = jnp.full(
            (max_layers, self.buffer_size, self.max_fan_out),
            fill_value=-1,
            dtype=jnp.int32,
        )
        self.outgoing_conn_indices = jnp.full(
            (max_layers, self.buffer_size, self.max_fan_out),
            fill_value=-1,
            dtype=jnp.int32,
        )

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Forward pass through the dynamic network.

        Args:
            x: Input array of shape (input_dim,).

        Returns:
            (output, buffer) where output has shape (output_dim,) and buffer
            has shape (buffer_size,) containing all activation values (useful
            for the structure tracker to compute connection utilities).
        """
        fwd = _make_dynamic_forward(
            self.activation_fns, self.activation_deriv_fns,
            self.input_dim, self.max_layers,
            self.max_units_per_layer, self.buffer_size,
        )
        return fwd(
            self.weights, self.output_weights, x,
            self.input_indices, self.unit_mask,
            self.activation_indices, self.output_mask,
            self.outgoing_unit_indices, self.outgoing_conn_indices,
        )


def _build_outgoing_for_layer(
    input_indices_l: Array,
    max_units_per_layer: int,
    max_connections_per_unit: int,
    buffer_size: int,
    max_fan_out: int,
) -> Tuple[Array, Array]:
    """Build outgoing connection indices for a single layer (jittable).

    Args:
        input_indices_l: (max_units_per_layer, max_connections_per_unit) int32.
        max_units_per_layer: Number of unit slots.
        max_connections_per_unit: Number of connection slots per unit.
        buffer_size: Total buffer size.
        max_fan_out: Maximum outgoing connections per buffer position.

    Returns:
        (outgoing_unit_indices, outgoing_conn_indices) each of shape
        (buffer_size, max_fan_out).
    """
    n = max_units_per_layer * max_connections_per_unit
    flat_src = input_indices_l.reshape(-1)  # (n,)
    flat_u = jnp.repeat(jnp.arange(max_units_per_layer), max_connections_per_unit)
    flat_c = jnp.tile(jnp.arange(max_connections_per_unit), max_units_per_layer)

    active = flat_src >= 0
    sort_key = jnp.where(active, flat_src, buffer_size)

    order = jnp.argsort(sort_key, stable=True)
    sorted_src = sort_key[order]
    sorted_u = flat_u[order]
    sorted_c = flat_c[order]
    sorted_active = sorted_src < buffer_size

    # Within-group index via associative scan of max over start positions
    starts = jnp.concatenate([
        jnp.array([True]),
        sorted_src[1:] != sorted_src[:-1],
    ])
    start_positions = jnp.where(starts, jnp.arange(n), 0)
    group_start = jax.lax.associative_scan(jnp.maximum, start_positions)
    within_group = jnp.arange(n) - group_start

    valid = sorted_active & (within_group < max_fan_out)

    # Scatter into flattened (buffer_size, max_fan_out) arrays
    flat_idx = sorted_src * max_fan_out + within_group
    flat_idx = jnp.where(valid, flat_idx, 0)

    out_u = jnp.full(buffer_size * max_fan_out, -1, dtype=jnp.int32)
    out_c = jnp.full(buffer_size * max_fan_out, -1, dtype=jnp.int32)
    out_u = out_u.at[flat_idx].set(jnp.where(valid, sorted_u, -1))
    out_c = out_c.at[flat_idx].set(jnp.where(valid, sorted_c, -1))

    return out_u.reshape(buffer_size, max_fan_out), out_c.reshape(buffer_size, max_fan_out)


def build_outgoing_indices(network: DynamicNetwork) -> DynamicNetwork:
    """Rebuild outgoing connection indices from input_indices (jittable).

    Must be called after any structural change (adding/removing connections).
    Returns a new network with updated outgoing arrays.
    """
    def per_layer(input_indices_l):
        return _build_outgoing_for_layer(
            input_indices_l,
            network.max_units_per_layer,
            network.max_connections_per_unit,
            network.buffer_size,
            network.max_fan_out,
        )

    out_u, out_c = jax.vmap(per_layer)(network.input_indices)

    return eqx.tree_at(
        lambda n: (n.outgoing_unit_indices, n.outgoing_conn_indices),
        network,
        (out_u, out_c),
    )


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
