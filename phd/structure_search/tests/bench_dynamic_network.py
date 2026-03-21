"""Speed comparison: DynamicNetwork vs dense MLP."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp

from phd.jax_core.models import MLP
from phd.structure_search.dynamic_network import (
    DynamicNetwork,
    build_outgoing_indices,
    count_active_connections,
    count_active_units,
    sync_outgoing_weights,
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


def make_sparse_dynamic(input_dim, output_dim, n_layers, hidden_dim, key,
                        max_connections_per_unit=64):
    """Create a DynamicNetwork with variable random connections per unit.

    Each unit gets a random number of connections drawn uniformly from
    [1, max_connections_per_unit]. The number of units per layer is chosen
    so the total connections approximately match an equivalent dense MLP.
    All hidden units in the last layer are connected to every output.
    """
    # MLP params: input->hidden + (n_layers-1)*hidden->hidden + hidden->output
    mlp_params = (input_dim * hidden_dim + (n_layers - 1) * hidden_dim * hidden_dim
                  + hidden_dim * output_dim)

    # Average connections per unit: E[Uniform(1, max)] for hidden + 1 output connection
    avg_conns_per_unit = (1 + max_connections_per_unit) / 2 + output_dim

    # Total units, split evenly across layers
    total_units = max(n_layers, round(mlp_params / avg_conns_per_unit))
    units_per_layer = max(1, total_units // n_layers)
    max_units_per_layer = units_per_layer

    net = DynamicNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        max_layers=n_layers,
        max_units_per_layer=max_units_per_layer,
        max_connections_per_unit=max_connections_per_unit,
        activations=('relu',),
        max_fan_out=max_connections_per_unit,
        key=key,
    )

    # Activate all units in each layer
    unit_mask = jnp.zeros_like(net.unit_mask)
    for l in range(n_layers):
        unit_mask = unit_mask.at[l, :units_per_layer].set(1)
    net = eqx.tree_at(lambda n: n.unit_mask, net, unit_mask)

    # Wire up connections with variable fan-in per unit
    for l in range(n_layers):
        n_units = units_per_layer

        # Available sources: previous layer only
        if l == 0:
            available_sources = jnp.arange(input_dim)
        else:
            prev_offset = input_dim + (l - 1) * max_units_per_layer
            available_sources = jnp.arange(prev_offset, prev_offset + units_per_layer)
        n_available = available_sources.shape[0]

        # Random keys for each unit
        conn_keys = jax.random.split(jax.random.fold_in(key, l * 1000), n_units)
        weight_keys = jax.random.split(jax.random.fold_in(key, l * 2000), n_units)

        # Sample random number of connections per unit: Uniform(1, max_connections)
        # Clamp to n_available if the previous layer is small
        max_possible = min(max_connections_per_unit, n_available)
        n_conns_per_unit = jax.random.randint(
            jax.random.fold_in(key, l * 3000),
            shape=(n_units,), minval=1, maxval=max_possible + 1,
        )

        # For each unit: sample n_conns sources, pad rest with -1
        def sample_unit(conn_key, w_key, n_conns):
            # Sample max_possible indices, then mask to n_conns
            perm = jax.random.permutation(conn_key, n_available)[:max_possible]
            sources = available_sources[perm]
            sources = jnp.sort(sources)
            # Mask: keep first n_conns, rest become -1
            slot_idx = jnp.arange(max_connections_per_unit)
            idx = jnp.where(slot_idx < n_conns, sources[slot_idx % max_possible], -1)
            # Weights
            w = jax.random.normal(w_key, (max_connections_per_unit,)) * 0.01
            w = jnp.where(slot_idx < n_conns, w, 0.0)
            return idx, w

        all_idx, all_w = jax.vmap(sample_unit)(conn_keys, weight_keys, n_conns_per_unit)

        # Place into full arrays
        idx = jnp.full((max_units_per_layer, max_connections_per_unit), -1, dtype=jnp.int32)
        idx = idx.at[:n_units].set(all_idx)
        w = jnp.zeros((max_units_per_layer, max_connections_per_unit), dtype=jnp.float32)
        w = w.at[:n_units].set(all_w)

        net = eqx.tree_at(lambda n: n.input_indices, net, net.input_indices.at[l].set(idx))
        net = eqx.tree_at(lambda n: n.weights, net, net.weights.at[l].set(w))

    # Connect all hidden units in last layer to every output
    if n_layers > 0:
        last_offset = input_dim + (n_layers - 1) * max_units_per_layer
        out_mask = net.output_mask.at[:, last_offset:last_offset + units_per_layer].set(1)
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
    dyn_sparse = make_sparse_dynamic(input_dim, output_dim, n_layers, hidden_dim, key, max_connections_per_unit=64)

    # Model stats
    def count_allocated(model):
        """Total allocated float array capacity (includes padding)."""
        return sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))

    mlp_hidden = sum(l.weight.shape[0] for l in mlp.layers[:-1])
    mlp_params = count_allocated(mlp)

    dyn_hidden = count_active_units(dyn)
    dyn_conns = count_active_connections(dyn)
    dyn_weights = dyn.weights.size + dyn.output_weights.size
    dyn_alloc = count_allocated(dyn)

    dyn_sparse_hidden = count_active_units(dyn_sparse)
    dyn_sparse_conns = count_active_connections(dyn_sparse)
    dyn_sparse_weights = dyn_sparse.weights.size + dyn_sparse.output_weights.size
    dyn_sparse_alloc = count_allocated(dyn_sparse)

    print(f"\nModel stats:")
    print(f"  {'MLP':<24s} {mlp_hidden:>6} hidden units, {mlp_params:>10,} params")
    print(f"  {'DynamicNet (dense)':<24s} {dyn_hidden:>6} hidden units, {dyn_conns:>10,} connections, {dyn_weights:>10,} weights, {dyn_alloc:>10,} allocated")
    print(f"  {'DynamicNet (sparse)':<24s} {dyn_sparse_hidden:>6} hidden units, {dyn_sparse_conns:>10,} connections, {dyn_sparse_weights:>10,} weights, {dyn_sparse_alloc:>10,} allocated")

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
    def loss_fn(model, x):
        out, _ = jax.vmap(model)(x)
        return jnp.mean(out ** 2)

    mlp_grad_fn = jax.jit(lambda x: eqx.filter_value_and_grad(loss_fn)(mlp, x))
    dyn_grad_fn = jax.jit(lambda x: eqx.filter_value_and_grad(loss_fn)(dyn, x))
    dyn_sparse_grad_fn = jax.jit(lambda x: eqx.filter_value_and_grad(loss_fn)(dyn_sparse, x))

    mlp_grad_ms = bench(mlp_grad_fn, x)
    dyn_grad_ms = bench(dyn_grad_fn, x)
    dyn_sparse_grad_ms = bench(dyn_sparse_grad_fn, x)

    print(f"\nForward + backward:")
    print(fmt("MLP", mlp_grad_ms, 1.0))
    print(fmt("DynamicNet (dense)", dyn_grad_ms, dyn_grad_ms / mlp_grad_ms))
    print(fmt("DynamicNet (sparse)", dyn_sparse_grad_ms, dyn_sparse_grad_ms / mlp_grad_ms))

    # build_outgoing_indices standalone (includes sync)
    def bench_rebuild(net, warmup=5, iters=200):
        rebuild_fn = jax.jit(build_outgoing_indices)
        for _ in range(warmup):
            rebuild_fn(net)
        jax.block_until_ready(rebuild_fn(net).outgoing_weights)
        t0 = time.perf_counter()
        for _ in range(iters):
            result = rebuild_fn(net)
        jax.block_until_ready(result.outgoing_weights)
        return (time.perf_counter() - t0) / iters * 1000

    dyn_rebuild_ms = bench_rebuild(dyn)
    dyn_sparse_rebuild_ms = bench_rebuild(dyn_sparse)

    # sync_outgoing_weights standalone
    def bench_sync(net, warmup=5, iters=200):
        sync_fn = jax.jit(sync_outgoing_weights)
        for _ in range(warmup):
            sync_fn(net)
        jax.block_until_ready(sync_fn(net).outgoing_weights)
        t0 = time.perf_counter()
        for _ in range(iters):
            result = sync_fn(net)
        jax.block_until_ready(result.outgoing_weights)
        return (time.perf_counter() - t0) / iters * 1000

    dyn_sync_ms = bench_sync(dyn)
    dyn_sparse_sync_ms = bench_sync(dyn_sparse)

    print(f"\nSync outgoing weights:")
    print(fmt("DynamicNet (dense)", dyn_sync_ms, dyn_sync_ms / mlp_grad_ms))
    print(fmt("DynamicNet (sparse)", dyn_sparse_sync_ms, dyn_sparse_sync_ms / mlp_grad_ms))

    dyn_total = dyn_rebuild_ms + dyn_sync_ms + dyn_grad_ms
    dyn_sparse_total = dyn_sparse_rebuild_ms + dyn_sparse_sync_ms + dyn_sparse_grad_ms

    print(f"\nRebuild + sync + forward + backward:")
    print(fmt("MLP", mlp_grad_ms, 1.0))
    print(fmt("DynamicNet (dense)", dyn_total, dyn_total / mlp_grad_ms))
    print(fmt("DynamicNet (sparse)", dyn_sparse_total, dyn_sparse_total / mlp_grad_ms))

    # Per-step cost (sync only — rebuild is amortized over structure changes)
    dyn_step = dyn_sync_ms + dyn_grad_ms
    dyn_sparse_step = dyn_sparse_sync_ms + dyn_sparse_grad_ms

    print(f"\nSync + forward + backward (per-step cost):")
    print(fmt("MLP", mlp_grad_ms, 1.0))
    print(fmt("DynamicNet (dense)", dyn_step, dyn_step / mlp_grad_ms))
    print(fmt("DynamicNet (sparse)", dyn_sparse_step, dyn_sparse_step / mlp_grad_ms))


if __name__ == '__main__':
    device = jax.devices()[0].platform
    print(f"JAX device: {jax.devices()[0]}")

    # Small config (like a quick experiment)
    run_benchmark(input_dim=128, output_dim=10, n_layers=2, hidden_dim=64, batch_size=1)

    # Medium config (closer to actual CIFAR-10 use)
    run_benchmark(input_dim=3072, output_dim=10, n_layers=2, hidden_dim=128, batch_size=1)

    # Larger hidden dim
    run_benchmark(input_dim=3072, output_dim=10, n_layers=3, hidden_dim=256, batch_size=1)
