"""Tests for BlockSparseMLP and compute_hidden_dim_for_params."""

import jax
import jax.numpy as jnp
import numpy as np

from phd.structure_search.block_sparse_mlp import (
    BlockSparseMLP,
    compute_hidden_dim_for_params,
)
from phd.jax_core.utils import count_params


def test_output_shape():
    """Output shape should be (K * output_dim_per_task,)."""
    model = BlockSparseMLP(
        n_tasks=3, input_dim_per_task=784, output_dim_per_task=10,
        n_layers=3, hidden_dim=16, key=jax.random.key(0),
    )
    x = jnp.ones(3 * 784)
    output, param_inputs = model(x)
    assert output.shape == (3 * 10,), f'Expected (30,), got {output.shape}'
    assert len(param_inputs) == 3  # n_layers inputs
    print('PASS: test_output_shape')


def test_single_layer():
    """With n_layers=1, output is direct linear transform per block."""
    model = BlockSparseMLP(
        n_tasks=2, input_dim_per_task=4, output_dim_per_task=3,
        n_layers=1, hidden_dim=8,  # hidden_dim ignored for n_layers=1
        key=jax.random.key(42),
    )
    x = jnp.ones(2 * 4)
    output, _ = model(x)
    assert output.shape == (6,)
    print('PASS: test_single_layer')


def test_block_independence():
    """Zeroing one block's input should not affect other blocks' outputs."""
    model = BlockSparseMLP(
        n_tasks=3, input_dim_per_task=8, output_dim_per_task=2,
        n_layers=3, hidden_dim=4, key=jax.random.key(1),
    )
    x_full = jax.random.normal(jax.random.key(99), (3 * 8,))
    out_full, _ = model(x_full)

    # Zero out task 1 (middle block)
    x_zeroed = x_full.at[8:16].set(0.0)
    out_zeroed, _ = model(x_zeroed)

    # Task 0 and task 2 outputs should be unchanged
    np.testing.assert_allclose(out_full[:2], out_zeroed[:2], atol=1e-6)
    np.testing.assert_allclose(out_full[4:], out_zeroed[4:], atol=1e-6)
    # Task 1 output should differ
    assert not np.allclose(out_full[2:4], out_zeroed[2:4])
    print('PASS: test_block_independence')


def test_param_count():
    """Parameter count should match the analytical formula."""
    K, h, n_layers = 5, 16, 3
    model = BlockSparseMLP(
        n_tasks=K, input_dim_per_task=784, output_dim_per_task=10,
        n_layers=n_layers, hidden_dim=h, key=jax.random.key(0),
    )
    expected = K * (784 * h + (n_layers - 2) * h * h + h * 10)
    actual = count_params(model)
    assert actual == expected, f'Expected {expected}, got {actual}'
    print('PASS: test_param_count')


def test_vmap_compatible():
    """Model should work under jax.vmap (batched forward pass)."""
    model = BlockSparseMLP(
        n_tasks=2, input_dim_per_task=8, output_dim_per_task=3,
        n_layers=2, hidden_dim=4, key=jax.random.key(0),
    )
    batch = jax.random.normal(jax.random.key(1), (32, 2 * 8))
    outputs, _ = jax.vmap(model)(batch)
    assert outputs.shape == (32, 6)
    print('PASS: test_vmap_compatible')


def test_compute_hidden_dim_block_sparse():
    """compute_hidden_dim_for_params should approximate the target."""
    target = 100_000
    h = compute_hidden_dim_for_params(target, 'block_sparse', 3, 10)
    actual = 10 * (784 * h + h * h + 10 * h)
    # Should be close to target (within one hidden unit's worth of params)
    assert actual <= target, f'hidden_dim={h} gives {actual} > {target}'
    h_plus = h + 1
    actual_plus = 10 * (784 * h_plus + h_plus * h_plus + 10 * h_plus)
    assert actual_plus > target, f'hidden_dim={h+1} should exceed target'
    print(f'PASS: test_compute_hidden_dim_block_sparse (h={h}, params={actual})')


def test_compute_hidden_dim_dense():
    """compute_hidden_dim_for_params for dense MLP should approximate the target."""
    target = 100_000
    H = compute_hidden_dim_for_params(target, 'mlp', 3, 10)
    actual = 7840 * H + H * H + 100 * H
    assert actual <= target
    H_plus = H + 1
    actual_plus = 7840 * H_plus + H_plus * H_plus + 100 * H_plus
    assert actual_plus > target
    print(f'PASS: test_compute_hidden_dim_dense (H={H}, params={actual})')


def test_gradient_flows():
    """Gradients should be non-zero for all weight layers."""
    model = BlockSparseMLP(
        n_tasks=2, input_dim_per_task=8, output_dim_per_task=3,
        n_layers=3, hidden_dim=4, key=jax.random.key(0),
    )
    x = jax.random.normal(jax.random.key(1), (8 * 2,))

    def loss_fn(m):
        out, _ = m(x)
        return out.sum()

    grads = jax.grad(loss_fn)(model)
    for i, g in enumerate(grads.layers):
        assert jnp.abs(g).sum() > 0, f'Layer {i} has zero gradients'
    print('PASS: test_gradient_flows')


if __name__ == '__main__':
    test_output_shape()
    test_single_layer()
    test_block_independence()
    test_param_count()
    test_vmap_compatible()
    test_compute_hidden_dim_block_sparse()
    test_compute_hidden_dim_dense()
    test_gradient_flows()
    print('\nAll BlockSparseMLP tests passed!')
