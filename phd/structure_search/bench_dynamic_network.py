"""Speed comparison: DynamicNetwork vs dense MLP."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp

from functools import partial

from phd.jax_core.models import MLP
from phd.structure_search.dynamic_network import (
    DynamicNetwork,
    _build_outgoing_for_layer,
    build_outgoing_indices,
)


def make_fully_connected_dynamic(input_dim, output_dim, n_layers, hidden_dim, key):
    """Create a DynamicNetwork with all units active and fully connected,
    matching an equivalent dense MLP configuration."""
    net = DynamicNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=n_layers,
        max_units_per_layer=hidden_dim,
        max_connections_per_unit=max(input_dim, hidden_dim),
        activations=('relu',),
        max_fan_out=hidden_dim,
        key=key,
    )

    # Activate all hidden units
    net = eqx.tree_at(lambda n: n.unit_mask, net, jnp.ones_like(net.unit_mask))

    # Wire up each layer fully
    for l in range(n_layers):
        if l == 0:
            src_dim = input_dim
            src_offset = 0
        else:
            src_dim = hidden_dim
            src_offset = input_dim + (l - 1) * hidden_dim

        # Build index array: each unit connects to all src_dim inputs, rest padded -1
        idx = jnp.full((hidden_dim, net.max_connections_per_unit), -1, dtype=jnp.int32)
        src_indices = jnp.arange(src_dim) + src_offset
        idx = idx.at[:, :src_dim].set(src_indices[None, :])
        net = eqx.tree_at(
            lambda n: n.input_indices,
            net,
            net.input_indices.at[l].set(idx),
        )

        # Random weights for active connections
        w_key = jax.random.fold_in(key, l)
        w = jax.random.normal(w_key, (hidden_dim, net.max_connections_per_unit)) * 0.01
        w = w.at[:, src_dim:].set(0.0)
        net = eqx.tree_at(lambda n: n.weights, net, net.weights.at[l].set(w))

    # Wire all hidden units in last layer to outputs
    last_offset = input_dim + (n_layers - 1) * hidden_dim
    out_mask = net.output_mask.at[:, last_offset:last_offset + hidden_dim].set(1)
    net = eqx.tree_at(lambda n: n.output_mask, net, out_mask)

    return build_outgoing_indices(net)


def make_sparse_dynamic(input_dim, output_dim, n_layers, hidden_dim, key, connections_per_unit=32):
    """Create a DynamicNetwork with 32 random connections per hidden unit,
    matching the total number of connections in an equivalent dense MLP.
    
    This requires more hidden units than the dense network to compensate for
    the reduced connections per unit.
    """
    # Calculate total hidden connections in dense MLP
    # Input to first hidden: input_dim * hidden_dim
    # Hidden to hidden: (n_layers - 1) * hidden_dim * hidden_dim
    dense_hidden_connections = input_dim * hidden_dim + (n_layers - 1) * hidden_dim * hidden_dim
    
    # Calculate how many hidden units we need to match this
    total_hidden_units_needed = (dense_hidden_connections + connections_per_unit - 1) // connections_per_unit
    
    # Distribute units across layers proportionally
    # First layer gets input_dim * hidden_dim / connections_per_unit units
    first_layer_units = (input_dim * hidden_dim + connections_per_unit - 1) // connections_per_unit
    remaining_units = total_hidden_units_needed - first_layer_units
    
    # Remaining layers share the rest (distribute evenly)
    if n_layers > 1:
        units_per_remaining_layer = (remaining_units + n_layers - 2) // (n_layers - 1)
    else:
        units_per_remaining_layer = 0
    
    # Calculate max units per layer needed
    max_units_per_layer = max(first_layer_units, units_per_remaining_layer, hidden_dim)
    
    # max_fan_out: conservative estimate for random connectivity.
    # Expected fan-out per source ≈ total_conns / n_sources.  Use 2x for headroom.
    max_total_conns = max_units_per_layer * connections_per_unit
    min_sources = min(input_dim, max_units_per_layer)
    estimated_max_fan = min(2 * max_total_conns // max(min_sources, 1), max_units_per_layer)
    estimated_max_fan = max(estimated_max_fan, connections_per_unit)

    # Create network with enough capacity
    net = DynamicNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=n_layers,
        max_units_per_layer=max_units_per_layer,
        max_connections_per_unit=connections_per_unit,
        activations=('relu',),
        max_fan_out=estimated_max_fan,
        key=key,
    )
    
    # Activate units in each layer
    unit_mask = jnp.zeros_like(net.unit_mask)
    if n_layers > 0:
        unit_mask = unit_mask.at[0, :first_layer_units].set(1)
        for l in range(1, n_layers):
            unit_mask = unit_mask.at[l, :units_per_remaining_layer].set(1)
    net = eqx.tree_at(lambda n: n.unit_mask, net, unit_mask)
    
    # Wire up connections randomly
    for l in range(n_layers):
        # Determine how many units are active in this layer
        if l == 0:
            n_units = first_layer_units
        else:
            n_units = units_per_remaining_layer
        
        # Get available source indices (match dense: each layer connects only to immediate previous layer)
        if l == 0:
            # First layer: only connect to inputs
            available_sources = jnp.arange(input_dim)
        else:
            # Later layers: connect only to previous layer (matching dense MLP structure)
            if l == 1:
                # Connect to first layer
                prev_offset = input_dim
                prev_units = first_layer_units
            else:
                # Connect to previous hidden layer
                prev_offset = input_dim + (l - 1) * max_units_per_layer
                prev_units = units_per_remaining_layer
            available_sources = jnp.arange(prev_offset, prev_offset + prev_units)
        
        # Randomly select connections_per_unit connections for each unit (vectorized)
        n_available = available_sources.shape[0]
        
        # Generate all random keys at once
        conn_keys = jax.random.split(jax.random.fold_in(key, l * 1000), n_units)
        weight_keys = jax.random.split(jax.random.fold_in(key, l * 2000), n_units)
        
        # Helper function to generate weights for a single unit
        def generate_weights(w_key, n_conns):
            return jax.random.normal(w_key, (n_conns,)) * 0.01
        
        if n_available <= connections_per_unit:
            # Use all available connections for all units (just repeat)
            selected_indices = jnp.tile(available_sources[None, :], (n_units, 1))
            # Pad with -1 to match connections_per_unit
            padding = jnp.full((n_units, connections_per_unit - n_available), -1, dtype=jnp.int32)
            idx_partial = jnp.concatenate([selected_indices, padding], axis=1)
            # Create full idx array and set the active units
            idx = jnp.full((max_units_per_layer, connections_per_unit), -1, dtype=jnp.int32)
            idx = idx.at[:n_units, :].set(idx_partial)
            # Generate weights for all units at once using vmap
            w_vals = jax.vmap(lambda k: generate_weights(k, n_available))(weight_keys)
            w_padding = jnp.zeros((n_units, connections_per_unit - n_available), dtype=jnp.float32)
            w_partial = jnp.concatenate([w_vals, w_padding], axis=1)
            w = jnp.zeros((max_units_per_layer, connections_per_unit), dtype=jnp.float32)
            w = w.at[:n_units, :].set(w_partial)
        else:
            # Vectorized random sampling for all units
            def sample_connections(conn_key):
                selected = jax.random.choice(
                    conn_key, available_sources, shape=(connections_per_unit,), replace=False
                )
                return jnp.sort(selected)
            
            # Use vmap to sample for all units at once
            selected_indices = jax.vmap(sample_connections)(conn_keys)
            idx = jnp.full((max_units_per_layer, connections_per_unit), -1, dtype=jnp.int32)
            idx = idx.at[:n_units, :].set(selected_indices)
            
            # Generate weights for all units at once using vmap
            w_vals = jax.vmap(lambda k: generate_weights(k, connections_per_unit))(weight_keys)
            w = jnp.zeros((max_units_per_layer, connections_per_unit), dtype=jnp.float32)
            w = w.at[:n_units, :].set(w_vals)
        
        net = eqx.tree_at(
            lambda n: n.input_indices,
            net,
            net.input_indices.at[l].set(idx),
        )
        net = eqx.tree_at(lambda n: n.weights, net, net.weights.at[l].set(w))
    
    # Wire last layer units to outputs (match dense: hidden_dim units to all outputs)
    # Dense has hidden_dim * output_dim output connections
    # We connect hidden_dim units from last layer to all outputs to match this
    if n_layers > 0:
        last_offset = input_dim + (n_layers - 1) * max_units_per_layer
        # Connect only hidden_dim units to match dense network's output connections
        out_mask = net.output_mask.at[:, last_offset:last_offset + hidden_dim].set(1)
        net = eqx.tree_at(lambda n: n.output_mask, net, out_mask)

    return build_outgoing_indices(net)


def bench(fn, x, warmup=5, iters=200):
    """Time a jitted function. Returns mean ms per call."""
    for _ in range(warmup):
        _ = fn(x)
    jax.block_until_ready(fn(x))

    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn(x)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000


def bench_grad(fn, x, warmup=5, iters=200):
    """Time a jitted grad function. Returns mean ms per call."""
    for _ in range(warmup):
        _ = fn(x)
    jax.block_until_ready(fn(x))

    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn(x)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000


def run_benchmark(input_dim, output_dim, n_layers, hidden_dim, batch_size):
    key = jax.random.key(0)
    print(f"\n{'='*60}")
    print(f"Config: input={input_dim}, hidden={hidden_dim}, layers={n_layers}, "
          f"output={output_dim}, batch={batch_size}")
    print(f"{'='*60}")

    # Dense MLP
    mlp = MLP(
        input_dim=input_dim, output_dim=output_dim,
        n_layers=n_layers + 1,  # MLP counts output layer
        hidden_dim=hidden_dim,
        weight_init_method='lecun_uniform',
        activation='relu',
        key=key,
    )

    # Fully-connected DynamicNetwork
    dyn = make_fully_connected_dynamic(input_dim, output_dim, n_layers, hidden_dim, key)

    # Sparse DynamicNetwork (32 connections per unit)
    dyn_sparse = make_sparse_dynamic(input_dim, output_dim, n_layers, hidden_dim, key, connections_per_unit=32)

    x = jax.random.normal(jax.random.key(1), (batch_size, input_dim))

    # Forward pass
    mlp_fwd = jax.jit(jax.vmap(mlp))
    dyn_fwd = jax.jit(jax.vmap(dyn))
    dyn_sparse_fwd = jax.jit(jax.vmap(dyn_sparse))

    mlp_ms = bench(mlp_fwd, x)
    dyn_ms = bench(dyn_fwd, x)
    dyn_sparse_ms = bench(dyn_sparse_fwd, x)

    def fmt(label, ms, ratio):
        return f"  {label:<24s} {ms:>8.3f} ms  ({ratio:>5.2f}x)"

    print(f"\nForward:")
    print(fmt("MLP", mlp_ms, 1.0))
    print(fmt("DynamicNet (dense)", dyn_ms, dyn_ms / mlp_ms))
    print(fmt("DynamicNet (sparse)", dyn_sparse_ms, dyn_sparse_ms / mlp_ms))

    # Forward + backward
    def mlp_loss(model, x):
        out, _ = jax.vmap(model)(x)
        return jnp.mean(out ** 2)

    def dyn_loss(model, x):
        out, _ = jax.vmap(model)(x)
        return jnp.mean(out ** 2)

    mlp_grad_fn = jax.jit(lambda x: eqx.filter_value_and_grad(mlp_loss)(mlp, x))
    dyn_grad_fn = jax.jit(lambda x: eqx.filter_value_and_grad(dyn_loss)(dyn, x))
    dyn_sparse_grad_fn = jax.jit(lambda x: eqx.filter_value_and_grad(dyn_loss)(dyn_sparse, x))

    mlp_grad_ms = bench_grad(mlp_grad_fn, x)
    dyn_grad_ms = bench_grad(dyn_grad_fn, x)
    dyn_sparse_grad_ms = bench_grad(dyn_sparse_grad_fn, x)

    print(f"\nForward + backward:")
    print(fmt("MLP", mlp_grad_ms, 1.0))
    print(fmt("DynamicNet (dense)", dyn_grad_ms, dyn_grad_ms / mlp_grad_ms))
    print(fmt("DynamicNet (sparse)", dyn_sparse_grad_ms, dyn_sparse_grad_ms / mlp_grad_ms))

    # build_outgoing_indices standalone (with input_indices as dynamic/traced)
    def make_rebuild_fn(net):
        @partial(jax.jit, static_argnums=(1, 2, 3, 4))
        def rebuild_fn(input_indices, max_units, max_conns, buf_size, max_fan):
            def per_layer(input_indices_l):
                return _build_outgoing_for_layer(
                    input_indices_l, max_units, max_conns, buf_size, max_fan,
                )
            return jax.vmap(per_layer)(input_indices)
        return rebuild_fn

    def bench_rebuild(net, warmup=5, iters=200):
        rebuild_fn = make_rebuild_fn(net)
        args = (net.input_indices, net.max_units_per_layer,
                net.max_connections_per_unit, net.buffer_size, net.max_fan_out)
        for _ in range(warmup):
            rebuild_fn(*args)
        jax.block_until_ready(rebuild_fn(*args))
        t0 = time.perf_counter()
        for _ in range(iters):
            result = rebuild_fn(*args)
        jax.block_until_ready(result)
        return (time.perf_counter() - t0) / iters * 1000

    dyn_rebuild_ms = bench_rebuild(dyn)
    dyn_sparse_rebuild_ms = bench_rebuild(dyn_sparse)

    dyn_total = dyn_rebuild_ms + dyn_grad_ms
    dyn_sparse_total = dyn_sparse_rebuild_ms + dyn_sparse_grad_ms

    print(f"\nRebuild + forward + backward:")
    print(fmt("MLP", mlp_grad_ms, 1.0))
    print(fmt("DynamicNet (dense)", dyn_total, dyn_total / mlp_grad_ms))
    print(fmt("DynamicNet (sparse)", dyn_sparse_total, dyn_sparse_total / mlp_grad_ms))


if __name__ == '__main__':
    device = jax.devices()[0].platform
    print(f"JAX device: {jax.devices()[0]}")

    # Small config (like a quick experiment)
    run_benchmark(input_dim=128, output_dim=10, n_layers=2, hidden_dim=64, batch_size=1)

    # Medium config (closer to actual CIFAR-10 use)
    run_benchmark(input_dim=3072, output_dim=10, n_layers=2, hidden_dim=128, batch_size=1)

    # Larger hidden dim
    run_benchmark(input_dim=3072, output_dim=10, n_layers=3, hidden_dim=256, batch_size=1)
