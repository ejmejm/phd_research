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
                          input_dim, max_layers, max_units_per_layer, buffer_size,
                          input_indices, unit_mask, activation_indices, output_mask,
                          outgoing_unit_indices, outgoing_weights):
    """Create a custom_vjp forward function with config and structure bound.

    Static Python values (functions, ints) and int32 structure arrays are
    captured in the closure.  Only differentiable JAX arrays (weights,
    output_weights, x) are passed as arguments to the custom_vjp function,
    so the backward returns only 3 cotangents instead of 9.

    outgoing_weights is float32 but captured in the closure (not an argument),
    so it receives no gradients. It must be synced from incoming weights via
    sync_outgoing_weights() after each optimizer step.
    """

    @jax.custom_vjp
    def dynamic_forward(weights, output_weights, x):
        output, buffer, _pre_acts = _forward_impl(
            weights, output_weights, x,
            input_indices, unit_mask, activation_indices, output_mask,
            activation_fns, activation_deriv_fns,
            input_dim, max_layers, max_units_per_layer, buffer_size,
        )
        return output, buffer

    def dynamic_forward_fwd(weights, output_weights, x):
        output, buffer, pre_acts = _forward_impl(
            weights, output_weights, x,
            input_indices, unit_mask, activation_indices, output_mask,
            activation_fns, activation_deriv_fns,
            input_dim, max_layers, max_units_per_layer, buffer_size,
        )
        residuals = (buffer, pre_acts, weights, output_weights)
        return (output, buffer), residuals

    def dynamic_forward_bwd(residuals, g):
        buffer, pre_acts, weights, output_weights = residuals
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

            # 4. Buffer gradient propagation (gather from pre-arranged outgoing weights)
            out_u = outgoing_unit_indices[l]  # (buffer_size, max_fan_out)
            out_mask = (out_u >= 0).astype(jnp.float32)
            safe_u = jnp.maximum(out_u, 0)
            contrib = grad_pre_act[safe_u] * outgoing_weights[l] * out_mask
            grad_buffer = grad_buffer + contrib.sum(axis=-1)

        # 5. Input gradient
        grad_x = grad_buffer[:input_dim]

        return (grad_weights, grad_output_weights, grad_x)

    dynamic_forward.defvjp(dynamic_forward_fwd, dynamic_forward_bwd)
    return dynamic_forward


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
    outgoing_weights: Array      # (max_layers, buffer_size, max_fan_out)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_layers: int,
        max_units_per_layer: int,
        max_connections_per_unit: int,
        activations: Tuple[str, ...] = ('relu',),
        max_fan_out: int | None = None,
        init_strategy: str = 'linear',
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the dynamic network.

        Args:
            input_dim: Input dimensionality (e.g. 3072 for flattened CIFAR-10).
            output_dim: Number of outputs (e.g. 10 for CIFAR-10).
            max_layers: Maximum number of hidden layers.
            max_units_per_layer: Maximum hidden units per layer.
            max_connections_per_unit: Maximum incoming connections per hidden unit.
            activations: Tuple of activation function names from ACTIVATION_MAP.
            max_fan_out: Maximum outgoing connections per buffer position per
                layer. Defaults to max_units_per_layer (fully-connected worst case).
            init_strategy: 'linear' (input→output with lecun_uniform weights)
                or 'empty' (no connections, network outputs zeros).
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

        # Output layer
        if init_strategy == 'linear':
            # Linear model: lecun_uniform input-to-output
            output_w = jnp.zeros((output_dim, self.buffer_size), dtype=jnp.float32)
            output_w = output_w.at[:, :input_dim].set(
                lecun_uniform(key, (output_dim, input_dim), in_dim=input_dim)
            )
            self.output_weights = output_w
        elif init_strategy == 'empty':
            # No connections — outputs zeros until units are generated
            self.output_weights = jnp.zeros(
                (output_dim, self.buffer_size), dtype=jnp.float32)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

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

        # Output mask
        if init_strategy == 'linear':
            output_m = jnp.zeros((output_dim, self.buffer_size), dtype=jnp.int32)
            output_m = output_m.at[:, :input_dim].set(1)
            self.output_mask = output_m
        else:
            # 'empty': no connections
            self.output_mask = jnp.zeros(
                (output_dim, self.buffer_size), dtype=jnp.int32)

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
        self.outgoing_weights = jnp.zeros(
            (max_layers, self.buffer_size, self.max_fan_out),
            dtype=jnp.float32,
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
            self.input_indices, self.unit_mask,
            self.activation_indices, self.output_mask,
            self.outgoing_unit_indices,
            jax.lax.stop_gradient(self.outgoing_weights),
        )
        return fwd(self.weights, self.output_weights, x)


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

    # Scatter into flattened (buffer_size, max_fan_out) arrays.
    # Use a trash slot at the end so invalid entries don't overwrite valid ones
    # (JAX .at[].set with duplicate indices uses last-writer-wins).
    out_size = buffer_size * max_fan_out
    flat_idx = sorted_src * max_fan_out + within_group
    flat_idx = jnp.where(valid, flat_idx, out_size)  # invalid → trash slot

    out_u = jnp.full(out_size + 1, -1, dtype=jnp.int32)
    out_c = jnp.full(out_size + 1, -1, dtype=jnp.int32)
    out_u = out_u.at[flat_idx].set(jnp.where(valid, sorted_u, -1))
    out_c = out_c.at[flat_idx].set(jnp.where(valid, sorted_c, -1))

    return out_u[:out_size].reshape(buffer_size, max_fan_out), out_c[:out_size].reshape(buffer_size, max_fan_out)


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

    network = eqx.tree_at(
        lambda n: (n.outgoing_unit_indices, n.outgoing_conn_indices),
        network,
        (out_u, out_c),
    )
    return sync_outgoing_weights(network)


def sync_outgoing_weights(network: DynamicNetwork) -> DynamicNetwork:
    """Sync outgoing weights from incoming weights using outgoing index mapping.

    Must be called after any weight update (optimizer step) or structural change.
    Returns a new network with updated outgoing_weights array.
    """
    out_u = network.outgoing_unit_indices  # (L, buf, fan_out)
    out_c = network.outgoing_conn_indices  # (L, buf, fan_out)
    mask = (out_u >= 0).astype(jnp.float32)
    safe_u = jnp.maximum(out_u, 0)
    safe_c = jnp.maximum(out_c, 0)

    def per_layer(w_l, u_l, c_l, m_l):
        return w_l[u_l, c_l] * m_l

    new_ow = jax.vmap(per_layer)(network.weights, safe_u, safe_c, mask)
    return eqx.tree_at(lambda n: n.outgoing_weights, network, new_ow)


def count_active_connections(network: DynamicNetwork) -> int:
    """Count the total number of active connections in the network.

    Includes both hidden-layer connections (input_indices >= 0) and
    output-layer connections (output_mask == 1).
    """
    hidden = jnp.sum(network.input_indices >= 0)
    output = jnp.sum(network.output_mask)
    return hidden + output


def count_active_units(network: DynamicNetwork) -> int:
    """Count the total number of active hidden units."""
    return jnp.sum(network.unit_mask)


def init_random_dynamic_network(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    units_per_layer: int,
    max_units_per_layer: int | None = None,
    max_connections_per_unit: int | None = None,
    activations: Tuple[str, ...] = ('relu',),
    max_fan_out: int | None = None,
    connect_all_to_output: bool = False,
    init_strategy: str = 'linear',
    *,
    key: PRNGKeyArray,
) -> DynamicNetwork:
    """Create a DynamicNetwork with random connectivity.

    Each hidden unit connects to max_connections_per_unit randomly chosen
    positions from all prior layers (inputs + preceding hidden layers).

    Args:
        connect_all_to_output: If True, all active hidden units across all
            layers connect to the output. If False (default), only the last
            hidden layer connects to the output.
        init_strategy: 'linear' (input→output) or 'empty' (no connections).
            Only affects the units_per_layer=0 path; when units_per_layer > 0,
            output arrays are overwritten by the wiring logic below.
    """
    if max_units_per_layer is None:
        max_units_per_layer = units_per_layer
    if max_connections_per_unit is None:
        max_connections_per_unit = input_dim

    assert units_per_layer <= max_units_per_layer

    net = DynamicNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=n_layers,
        max_units_per_layer=max_units_per_layer,
        max_connections_per_unit=max_connections_per_unit,
        activations=activations,
        max_fan_out=max_fan_out,
        init_strategy=init_strategy,
        key=key,
    )

    if units_per_layer == 0:
        return build_outgoing_indices(net)

    buffer_size = net.buffer_size
    weights = net.weights
    input_indices = net.input_indices
    unit_mask = net.unit_mask

    for l in range(n_layers):
        # Build mask of available source positions (inputs + all prior layers)
        available_mask = jnp.zeros(buffer_size, dtype=jnp.bool_)
        available_mask = available_mask.at[:input_dim].set(True)
        for prev_l in range(l):
            offset = input_dim + prev_l * max_units_per_layer
            available_mask = available_mask.at[offset:offset + units_per_layer].set(True)

        n_available = int(available_mask.sum())
        n_conns = min(max_connections_per_unit, n_available)

        # Gather available indices into a dense array for permutation sampling
        available_indices = jnp.where(available_mask, jnp.arange(buffer_size), buffer_size)
        available_indices = jnp.sort(available_indices)[:n_available]  # (n_available,)

        # Sample random connections for all units at once via vmap
        key, layer_key = jax.random.split(key)
        unit_keys = jax.random.split(layer_key, units_per_layer)

        def sample_unit(unit_key):
            perm = jax.random.permutation(unit_key, n_available)[:n_conns]
            sources = jnp.sort(available_indices[perm])
            # Pad to max_connections_per_unit with -1
            padded = jnp.full(max_connections_per_unit, -1, dtype=jnp.int32)
            padded = padded.at[:n_conns].set(sources)
            return padded

        layer_indices = jax.vmap(sample_unit)(unit_keys)  # (units_per_layer, max_conns)

        # Place into full arrays (pad inactive units with -1)
        full_indices = jnp.full(
            (max_units_per_layer, max_connections_per_unit), -1, dtype=jnp.int32)
        full_indices = full_indices.at[:units_per_layer].set(layer_indices)
        input_indices = input_indices.at[l].set(full_indices)

        # Lecun uniform weights for active connections
        key, w_key = jax.random.split(key)
        w = lecun_uniform(w_key, (units_per_layer, max_connections_per_unit),
                          in_dim=n_conns)
        # Zero out padding slots
        conn_mask = (layer_indices >= 0).astype(jnp.float32)
        w = w * conn_mask
        full_w = jnp.zeros((max_units_per_layer, max_connections_per_unit))
        full_w = full_w.at[:units_per_layer].set(w)
        weights = weights.at[l].set(full_w)

        # Activate units
        unit_mask = unit_mask.at[l, :units_per_layer].set(1)

    # Output layer wiring
    output_mask = jnp.zeros_like(net.output_mask)
    output_weights = jnp.zeros_like(net.output_weights)

    if connect_all_to_output:
        # Connect all active hidden units across all layers to the output
        total_output_units = n_layers * units_per_layer
        key, ow_key = jax.random.split(key)
        for l in range(n_layers):
            offset = input_dim + l * max_units_per_layer
            output_mask = output_mask.at[:, offset:offset + units_per_layer].set(1)
            ow = lecun_uniform(
                jax.random.fold_in(ow_key, l),
                (output_dim, units_per_layer), in_dim=total_output_units,
            )
            output_weights = output_weights.at[:, offset:offset + units_per_layer].set(ow)
    else:
        # Connect only last hidden layer to the output
        last_offset = input_dim + (n_layers - 1) * max_units_per_layer
        output_mask = output_mask.at[:, last_offset:last_offset + units_per_layer].set(1)
        key, ow_key = jax.random.split(key)
        ow = lecun_uniform(ow_key, (output_dim, units_per_layer), in_dim=units_per_layer)
        output_weights = output_weights.at[:, last_offset:last_offset + units_per_layer].set(ow)

    net = eqx.tree_at(
        lambda n: (n.weights, n.input_indices, n.unit_mask,
                   n.output_mask, n.output_weights),
        net,
        (weights, input_indices, unit_mask, output_mask, output_weights),
    )

    return build_outgoing_indices(net)
