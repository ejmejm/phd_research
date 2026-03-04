"""Tests for DynamicNetwork."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phd.structure_search.dynamic_network import (
    DynamicNetwork,
    _dynamic_forward_plain,
    build_outgoing_indices,
    sync_outgoing_weights,
    count_active_connections,
    count_active_units,
)


def test_shapes():
    """All arrays should have the correct shapes after initialization."""
    net = DynamicNetwork(
        input_dim=8, output_dim=3, max_layers=2,
        max_units_per_layer=4, max_connections_per_unit=5,
        activations=('relu', 'tanh'), key=jax.random.key(0),
    )
    assert net.weights.shape == (2, 4, 5)
    assert net.output_weights.shape == (3, 8 + 2 * 4)  # (3, 16)
    assert net.input_indices.shape == (2, 4, 5)
    assert net.unit_mask.shape == (2, 4)
    assert net.activation_indices.shape == (2, 4)
    assert net.output_mask.shape == (3, 16)
    assert net.outgoing_weights.shape == (2, 16, 4)  # (max_layers, buffer_size, max_fan_out)
    assert net.buffer_size == 16
    print("PASS: test_shapes")


def test_linear_model_forward():
    """With no hidden units, output should equal output_weights[:, :input_dim] @ x."""
    input_dim, output_dim = 8, 3
    net = DynamicNetwork(
        input_dim=input_dim, output_dim=output_dim, max_layers=2,
        max_units_per_layer=4, max_connections_per_unit=5,
        activations=('relu',), key=jax.random.key(42),
    )

    x = jax.random.normal(jax.random.key(1), (input_dim,))
    output, buffer = net(x)

    expected = net.output_weights[:, :input_dim] @ x
    np.testing.assert_allclose(output, expected, atol=1e-6)

    # Buffer should have inputs in first slots and zeros elsewhere
    np.testing.assert_allclose(buffer[:input_dim], x, atol=1e-7)
    np.testing.assert_allclose(buffer[input_dim:], 0.0, atol=1e-7)
    print("PASS: test_linear_model_forward")


def test_single_hidden_unit():
    """Activate one hidden unit with known weights and verify the output."""
    input_dim, output_dim = 4, 2
    net = DynamicNetwork(
        input_dim=input_dim, output_dim=output_dim, max_layers=2,
        max_units_per_layer=3, max_connections_per_unit=4,
        activations=('relu', 'tanh'), key=jax.random.key(0),
    )

    # Activate unit 0 in layer 0, connecting to inputs 0 and 2
    net = eqx.tree_at(
        lambda n: n.unit_mask,
        net,
        net.unit_mask.at[0, 0].set(1),
    )
    net = eqx.tree_at(
        lambda n: n.input_indices,
        net,
        net.input_indices.at[0, 0, 0].set(0).at[0, 0, 1].set(2),
    )
    net = eqx.tree_at(
        lambda n: n.weights,
        net,
        net.weights.at[0, 0, 0].set(0.5).at[0, 0, 1].set(-0.3),
    )
    # Use relu (activation index 0 — already default)

    # Also set output connection for this hidden unit
    hidden_buffer_idx = input_dim  # layer 0, unit 0
    net = eqx.tree_at(
        lambda n: n.output_weights,
        net,
        net.output_weights.at[0, hidden_buffer_idx].set(1.0)
                         .at[1, hidden_buffer_idx].set(-0.5),
    )
    net = eqx.tree_at(
        lambda n: n.output_mask,
        net,
        net.output_mask.at[0, hidden_buffer_idx].set(1)
                       .at[1, hidden_buffer_idx].set(1),
    )

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    output, buffer = net(x)

    # Hidden unit 0: relu(x[0]*0.5 + x[2]*(-0.3)) = relu(0.5 - 0.9) = relu(-0.4) = 0.0
    expected_hidden = 0.0
    assert float(buffer[hidden_buffer_idx]) == expected_hidden

    # Output should be linear part + hidden contribution
    expected_output = net.output_weights[:, :input_dim] @ x  # linear part
    # hidden contribution is 0 because relu(-0.4)=0
    np.testing.assert_allclose(output, expected_output, atol=1e-6)

    # Now test with inputs that make relu positive
    x2 = jnp.array([2.0, 0.0, 1.0, 0.0])
    output2, buffer2 = net(x2)
    # relu(2.0*0.5 + 1.0*(-0.3)) = relu(1.0 - 0.3) = 0.7
    expected_hidden2 = 0.7
    np.testing.assert_allclose(buffer2[hidden_buffer_idx], expected_hidden2, atol=1e-6)

    expected_output2 = net.output_weights[:, :input_dim] @ x2
    expected_output2 = expected_output2 + jnp.array([1.0 * 0.7, -0.5 * 0.7])
    np.testing.assert_allclose(output2, expected_output2, atol=1e-6)
    print("PASS: test_single_hidden_unit")


def test_tanh_activation():
    """Verify tanh activation works via activation_indices."""
    input_dim, output_dim = 2, 1
    net = DynamicNetwork(
        input_dim=input_dim, output_dim=output_dim, max_layers=1,
        max_units_per_layer=2, max_connections_per_unit=2,
        activations=('relu', 'tanh'), key=jax.random.key(0),
    )

    # Activate unit 0 with tanh (activation index 1)
    net = eqx.tree_at(lambda n: n.unit_mask, net, net.unit_mask.at[0, 0].set(1))
    net = eqx.tree_at(lambda n: n.activation_indices, net, net.activation_indices.at[0, 0].set(1))
    net = eqx.tree_at(lambda n: n.input_indices, net, net.input_indices.at[0, 0, 0].set(0))
    net = eqx.tree_at(lambda n: n.weights, net, net.weights.at[0, 0, 0].set(1.0))

    hidden_idx = input_dim  # buffer index for layer 0, unit 0
    net = eqx.tree_at(lambda n: n.output_weights, net, net.output_weights.at[0, hidden_idx].set(1.0))
    net = eqx.tree_at(lambda n: n.output_mask, net, net.output_mask.at[0, hidden_idx].set(1))

    x = jnp.array([0.5, 0.0])
    output, buffer = net(x)

    expected_hidden = float(jnp.tanh(0.5))
    np.testing.assert_allclose(buffer[hidden_idx], expected_hidden, atol=1e-6)
    print("PASS: test_tanh_activation")


def test_gradient_only_for_weights():
    """eqx.filter_value_and_grad should only produce gradients for float arrays."""
    net = DynamicNetwork(
        input_dim=4, output_dim=2, max_layers=1,
        max_units_per_layer=3, max_connections_per_unit=4,
        activations=('relu',), key=jax.random.key(0),
    )

    x = jax.random.normal(jax.random.key(1), (4,))

    def loss_fn(model):
        output, _ = model(x)
        return jnp.sum(output ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(net)

    # Trainable (float32) fields should have gradients
    assert grads.weights is not None
    assert grads.output_weights is not None
    assert grads.weights.shape == net.weights.shape
    assert grads.output_weights.shape == net.output_weights.shape

    # Non-trainable (int32) fields should be None
    assert grads.input_indices is None
    assert grads.unit_mask is None
    assert grads.activation_indices is None
    assert grads.output_mask is None

    # outgoing_weights is float32 but stop_gradient'd — should be all zeros
    assert grads.outgoing_weights is not None  # float32 → not filtered out
    np.testing.assert_allclose(grads.outgoing_weights, 0.0, atol=1e-10)

    print("PASS: test_gradient_only_for_weights")


def test_vmap_batched_forward():
    """Verify vmap works for batched forward passes (as used in train.py)."""
    net = DynamicNetwork(
        input_dim=8, output_dim=3, max_layers=2,
        max_units_per_layer=4, max_connections_per_unit=5,
        activations=('relu',), key=jax.random.key(0),
    )

    batch = jax.random.normal(jax.random.key(1), (4, 8))
    outputs, buffers = jax.vmap(net)(batch)

    assert outputs.shape == (4, 3)
    assert buffers.shape == (4, net.buffer_size)
    print("PASS: test_vmap_batched_forward")


def test_multi_layer_buffer_propagation():
    """Unit in layer 1 should be able to read activations from layer 0."""
    input_dim, output_dim = 2, 1
    net = DynamicNetwork(
        input_dim=input_dim, output_dim=output_dim, max_layers=2,
        max_units_per_layer=2, max_connections_per_unit=2,
        activations=('relu',), key=jax.random.key(0),
    )

    # Layer 0, unit 0: reads input 0, weight 1.0
    net = eqx.tree_at(lambda n: n.unit_mask, net, net.unit_mask.at[0, 0].set(1))
    net = eqx.tree_at(lambda n: n.input_indices, net, net.input_indices.at[0, 0, 0].set(0))
    net = eqx.tree_at(lambda n: n.weights, net, net.weights.at[0, 0, 0].set(1.0))

    # Layer 1, unit 0: reads from layer 0 unit 0 (buffer index = input_dim + 0 = 2)
    l0_u0_idx = input_dim
    net = eqx.tree_at(lambda n: n.unit_mask, net, net.unit_mask.at[1, 0].set(1))
    net = eqx.tree_at(lambda n: n.input_indices, net, net.input_indices.at[1, 0, 0].set(l0_u0_idx))
    net = eqx.tree_at(lambda n: n.weights, net, net.weights.at[1, 0, 0].set(2.0))

    # Connect layer 1 unit 0 to output
    l1_u0_idx = input_dim + net.max_units_per_layer  # layer 1 offset
    net = eqx.tree_at(lambda n: n.output_weights, net, net.output_weights.at[0, l1_u0_idx].set(1.0))
    net = eqx.tree_at(lambda n: n.output_mask, net, net.output_mask.at[0, l1_u0_idx].set(1))

    x = jnp.array([3.0, 0.0])
    output, buffer = net(x)

    # Layer 0 unit 0: relu(3.0 * 1.0) = 3.0
    np.testing.assert_allclose(buffer[l0_u0_idx], 3.0, atol=1e-6)

    # Layer 1 unit 0: relu(3.0 * 2.0) = 6.0
    np.testing.assert_allclose(buffer[l1_u0_idx], 6.0, atol=1e-6)

    # Output: linear part + 6.0 * 1.0
    expected = float(net.output_weights[0, :input_dim] @ x) + 6.0
    np.testing.assert_allclose(output[0], expected, atol=1e-6)
    print("PASS: test_multi_layer_buffer_propagation")


def test_count_helpers():
    """Verify count_active_connections and count_active_units."""
    net = DynamicNetwork(
        input_dim=4, output_dim=2, max_layers=2,
        max_units_per_layer=3, max_connections_per_unit=4,
        activations=('relu',), key=jax.random.key(0),
    )

    # Initially: no hidden connections, but input-to-output connections are active
    assert count_active_units(net) == 0
    assert count_active_connections(net) == 4 * 2  # input_dim * output_dim

    # Activate a unit with 2 connections
    net = eqx.tree_at(lambda n: n.unit_mask, net, net.unit_mask.at[0, 0].set(1))
    net = eqx.tree_at(
        lambda n: n.input_indices,
        net,
        net.input_indices.at[0, 0, 0].set(0).at[0, 0, 1].set(1),
    )

    assert count_active_units(net) == 1
    assert count_active_connections(net) == 4 * 2 + 2  # output + hidden
    print("PASS: test_count_helpers")


def test_jit_compilation():
    """Verify the forward pass compiles and runs under jit."""
    net = DynamicNetwork(
        input_dim=8, output_dim=3, max_layers=2,
        max_units_per_layer=4, max_connections_per_unit=5,
        activations=('relu', 'tanh'), key=jax.random.key(0),
    )

    @jax.jit
    def forward(model, x):
        return model(x)

    x = jax.random.normal(jax.random.key(1), (8,))
    output, buffer = forward(net, x)
    assert output.shape == (3,)
    assert buffer.shape == (net.buffer_size,)

    # Run again to verify no recompilation issues
    x2 = jax.random.normal(jax.random.key(2), (8,))
    output2, buffer2 = forward(net, x2)
    assert output2.shape == (3,)
    print("PASS: test_jit_compilation")


def _build_test_network():
    """Create a small network with active units and connections for gradient tests."""
    input_dim, output_dim = 4, 2
    net = DynamicNetwork(
        input_dim=input_dim, output_dim=output_dim, max_layers=2,
        max_units_per_layer=3, max_connections_per_unit=4,
        activations=('relu',), key=jax.random.key(42),
    )

    # Layer 0: activate unit 0, connecting to inputs 0, 1, 2
    net = eqx.tree_at(lambda n: n.unit_mask, net, net.unit_mask.at[0, 0].set(1))
    net = eqx.tree_at(
        lambda n: n.input_indices,
        net,
        net.input_indices.at[0, 0, 0].set(0).at[0, 0, 1].set(1).at[0, 0, 2].set(2),
    )
    net = eqx.tree_at(
        lambda n: n.weights,
        net,
        net.weights.at[0, 0, 0].set(0.5).at[0, 0, 1].set(-0.3).at[0, 0, 2].set(0.7),
    )

    # Layer 1: activate unit 0, connecting to layer 0 unit 0 (buffer idx = input_dim)
    l0_u0_idx = input_dim
    net = eqx.tree_at(lambda n: n.unit_mask, net, net.unit_mask.at[1, 0].set(1))
    net = eqx.tree_at(
        lambda n: n.input_indices,
        net,
        net.input_indices.at[1, 0, 0].set(l0_u0_idx),
    )
    net = eqx.tree_at(
        lambda n: n.weights,
        net,
        net.weights.at[1, 0, 0].set(2.0),
    )

    # Connect layer 1 unit 0 to output
    l1_u0_idx = input_dim + net.max_units_per_layer
    net = eqx.tree_at(
        lambda n: n.output_weights,
        net,
        net.output_weights.at[0, l1_u0_idx].set(1.0).at[1, l1_u0_idx].set(-0.5),
    )
    net = eqx.tree_at(
        lambda n: n.output_mask,
        net,
        net.output_mask.at[0, l1_u0_idx].set(1).at[1, l1_u0_idx].set(1),
    )

    # Build outgoing indices
    net = build_outgoing_indices(net)
    return net


def test_build_outgoing_indices():
    """build_outgoing_indices should produce the correct reverse mapping."""
    net = _build_test_network()

    # Layer 0, unit 0 connects to inputs 0, 1, 2
    # So outgoing_unit_indices[0, 0, :] should contain unit 0 (for source position 0)
    out_u = net.outgoing_unit_indices
    out_c = net.outgoing_conn_indices

    # Check that input position 0 maps to (layer=0, unit=0, conn=0)
    found = False
    for f in range(net.max_fan_out):
        if int(out_u[0, 0, f]) == 0 and int(out_c[0, 0, f]) == 0:
            found = True
            break
    assert found, "Input 0 should map to layer 0 unit 0 conn 0"

    # Check that input position 1 maps to (layer=0, unit=0, conn=1)
    found = False
    for f in range(net.max_fan_out):
        if int(out_u[0, 1, f]) == 0 and int(out_c[0, 1, f]) == 1:
            found = True
            break
    assert found, "Input 1 should map to layer 0 unit 0 conn 1"

    # Layer 1, unit 0 reads from l0_u0_idx (=input_dim). Check outgoing.
    l0_u0_idx = net.input_dim
    found = False
    for f in range(net.max_fan_out):
        if int(out_u[1, l0_u0_idx, f]) == 0 and int(out_c[1, l0_u0_idx, f]) == 0:
            found = True
            break
    assert found, f"Buffer position {l0_u0_idx} should map to layer 1 unit 0 conn 0"

    print("PASS: test_build_outgoing_indices")


def test_sync_outgoing_weights():
    """sync_outgoing_weights should copy incoming weights into outgoing layout."""
    net = _build_test_network()

    # Layer 0, unit 0 has weights [0.5, -0.3, 0.7] at connections to inputs [0, 1, 2]
    # After sync, outgoing_weights[0, 0, :] should have 0.5 for the slot mapping to (unit=0, conn=0)
    out_u = net.outgoing_unit_indices
    out_c = net.outgoing_conn_indices
    ow = net.outgoing_weights

    # For each active outgoing slot, verify it matches the incoming weight
    for l in range(net.max_layers):
        for src in range(net.buffer_size):
            for f in range(net.max_fan_out):
                u = int(out_u[l, src, f])
                c = int(out_c[l, src, f])
                if u >= 0 and c >= 0:
                    expected = float(net.weights[l, u, c])
                    actual = float(ow[l, src, f])
                    np.testing.assert_allclose(
                        actual, expected, atol=1e-7,
                        err_msg=f"Outgoing weight mismatch at l={l} src={src} f={f} (unit={u} conn={c})"
                    )

    # Inactive slots should be zero
    inactive = (out_u < 0)
    np.testing.assert_allclose(ow[inactive], 0.0, atol=1e-10)

    print("PASS: test_sync_outgoing_weights")


def test_custom_vjp_matches_autodiff():
    """Custom backward should produce identical gradients to JAX autodiff."""
    net = _build_test_network()

    x = jax.random.normal(jax.random.key(1), (net.input_dim,))

    # Gradients via custom VJP (the new __call__)
    def loss_custom(model):
        output, _ = model(x)
        return jnp.sum(output ** 2)
    _, grads_custom = eqx.filter_value_and_grad(loss_custom)(net)

    # Gradients via plain autodiff (no custom VJP)
    def loss_plain(model):
        output, _ = _dynamic_forward_plain(
            model.weights, model.output_weights, x,
            model.input_indices, model.unit_mask,
            model.activation_indices, model.output_mask,
            model.activation_fns,
            model.input_dim, model.max_layers,
            model.max_units_per_layer, model.buffer_size,
        )
        return jnp.sum(output ** 2)
    _, grads_plain = eqx.filter_value_and_grad(loss_plain)(net)

    np.testing.assert_allclose(grads_custom.weights, grads_plain.weights, atol=1e-5)
    np.testing.assert_allclose(
        grads_custom.output_weights, grads_plain.output_weights, atol=1e-5
    )
    print("PASS: test_custom_vjp_matches_autodiff")


def test_gradient_finite_difference():
    """Gradients should match finite differences."""
    net = _build_test_network()
    x = jax.random.normal(jax.random.key(1), (net.input_dim,))

    def loss_fn(model):
        output, _ = model(x)
        return jnp.sum(output ** 2)

    _, grads = eqx.filter_value_and_grad(loss_fn)(net)

    eps = 1e-4
    # Check a few active weight positions
    for idx in [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0)]:
        w_plus = net.weights.at[idx].add(eps)
        w_minus = net.weights.at[idx].add(-eps)
        net_plus = eqx.tree_at(lambda n: n.weights, net, w_plus)
        net_minus = eqx.tree_at(lambda n: n.weights, net, w_minus)
        fd = (loss_fn(net_plus) - loss_fn(net_minus)) / (2 * eps)
        np.testing.assert_allclose(
            grads.weights[idx], fd, atol=1e-3,
            err_msg=f"Weight gradient mismatch at {idx}",
        )
    print("PASS: test_gradient_finite_difference")


def test_gradient_multi_layer():
    """Gradients should flow correctly through multiple layers."""
    net = _build_test_network()

    # Use an input that activates the relu paths
    x = jnp.array([2.0, 1.0, 0.5, 0.0])
    # Layer 0 unit 0: relu(2*0.5 + 1*(-0.3) + 0.5*0.7) = relu(1.05) = 1.05 > 0
    # Layer 1 unit 0: relu(1.05 * 2.0) = 2.1 > 0

    def loss_fn(model):
        output, _ = model(x)
        return jnp.sum(output ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(net)

    # Layer 1 weight should have non-zero gradient (it's on an active path)
    assert float(jnp.abs(grads.weights[1, 0, 0])) > 1e-6, \
        "Layer 1 weight should have non-zero gradient"

    # Layer 0 weights should also have non-zero gradient (gradient flows through layer 1)
    assert float(jnp.abs(grads.weights[0, 0, 0])) > 1e-6, \
        "Layer 0 weight should have non-zero gradient"

    # Compare with finite differences (use rtol for large gradients)
    eps = 1e-4
    for idx in [(0, 0, 0), (1, 0, 0)]:
        w_plus = net.weights.at[idx].add(eps)
        w_minus = net.weights.at[idx].add(-eps)
        net_plus = eqx.tree_at(lambda n: n.weights, net, w_plus)
        net_minus = eqx.tree_at(lambda n: n.weights, net, w_minus)
        fd = (loss_fn(net_plus) - loss_fn(net_minus)) / (2 * eps)
        np.testing.assert_allclose(
            grads.weights[idx], fd, rtol=1e-3,
            err_msg=f"Multi-layer gradient mismatch at {idx}",
        )
    print("PASS: test_gradient_multi_layer")


def test_gradient_multiple_activations():
    """Gradients should be correct with mixed activation functions."""
    input_dim, output_dim = 4, 2
    net = DynamicNetwork(
        input_dim=input_dim, output_dim=output_dim, max_layers=1,
        max_units_per_layer=3, max_connections_per_unit=4,
        activations=('relu', 'tanh', 'sigmoid'), key=jax.random.key(0),
    )

    # Unit 0: relu (index 0), Unit 1: tanh (index 1), Unit 2: sigmoid (index 2)
    net = eqx.tree_at(
        lambda n: n.unit_mask,
        net,
        net.unit_mask.at[0, 0].set(1).at[0, 1].set(1).at[0, 2].set(1),
    )
    net = eqx.tree_at(
        lambda n: n.activation_indices,
        net,
        net.activation_indices.at[0, 0].set(0).at[0, 1].set(1).at[0, 2].set(2),
    )
    # All units connect to input 0
    net = eqx.tree_at(
        lambda n: n.input_indices,
        net,
        net.input_indices.at[0, 0, 0].set(0).at[0, 1, 0].set(0).at[0, 2, 0].set(0),
    )
    net = eqx.tree_at(
        lambda n: n.weights,
        net,
        net.weights.at[0, 0, 0].set(0.5).at[0, 1, 0].set(-0.3).at[0, 2, 0].set(0.7),
    )

    # Connect all units to output
    for u in range(3):
        buf_idx = input_dim + u
        net = eqx.tree_at(
            lambda n: n.output_weights,
            net,
            net.output_weights.at[0, buf_idx].set(1.0),
        )
        net = eqx.tree_at(
            lambda n: n.output_mask,
            net,
            net.output_mask.at[0, buf_idx].set(1),
        )

    net = build_outgoing_indices(net)

    x = jax.random.normal(jax.random.key(1), (input_dim,))

    # Compare custom VJP vs autodiff
    def loss_custom(model):
        output, _ = model(x)
        return jnp.sum(output ** 2)

    def loss_plain(model):
        output, _ = _dynamic_forward_plain(
            model.weights, model.output_weights, x,
            model.input_indices, model.unit_mask,
            model.activation_indices, model.output_mask,
            model.activation_fns,
            model.input_dim, model.max_layers,
            model.max_units_per_layer, model.buffer_size,
        )
        return jnp.sum(output ** 2)

    _, grads_custom = eqx.filter_value_and_grad(loss_custom)(net)
    _, grads_plain = eqx.filter_value_and_grad(loss_plain)(net)

    np.testing.assert_allclose(grads_custom.weights, grads_plain.weights, atol=1e-5)
    np.testing.assert_allclose(
        grads_custom.output_weights, grads_plain.output_weights, atol=1e-5
    )
    print("PASS: test_gradient_multiple_activations")


def test_vmap_grad():
    """jax.vmap + eqx.filter_value_and_grad should work with custom VJP."""
    net = _build_test_network()
    batch = jax.random.normal(jax.random.key(1), (4, net.input_dim))

    def loss_fn(model):
        outputs, _ = jax.vmap(model)(batch)
        return jnp.mean(outputs ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
    assert grads.weights.shape == net.weights.shape
    assert not jnp.any(jnp.isnan(grads.weights))
    assert not jnp.any(jnp.isnan(grads.output_weights))
    print("PASS: test_vmap_grad")


def test_explicit_vjp_matches_optimized():
    """Explicit backward (dense J^T matmul) should match the optimized custom VJP."""
    net = _build_test_network()
    x = jax.random.normal(jax.random.key(1), (net.input_dim,))

    def loss_custom(model):
        output, _ = model(x)
        return jnp.sum(output ** 2)

    def loss_explicit(model):
        output, _ = model.forward_explicit(x)
        return jnp.sum(output ** 2)

    val_c, grads_custom = eqx.filter_value_and_grad(loss_custom)(net)
    val_e, grads_explicit = eqx.filter_value_and_grad(loss_explicit)(net)

    np.testing.assert_allclose(val_c, val_e, atol=1e-6)
    np.testing.assert_allclose(grads_explicit.weights, grads_custom.weights, atol=1e-5)
    np.testing.assert_allclose(
        grads_explicit.output_weights, grads_custom.output_weights, atol=1e-5
    )
    print("PASS: test_explicit_vjp_matches_optimized")


def test_explicit_vjp_matches_autodiff():
    """Explicit backward should match JAX's own autodiff (no custom VJP)."""
    net = _build_test_network()
    x = jax.random.normal(jax.random.key(1), (net.input_dim,))

    def loss_explicit(model):
        output, _ = model.forward_explicit(x)
        return jnp.sum(output ** 2)

    def loss_plain(model):
        output, _ = _dynamic_forward_plain(
            model.weights, model.output_weights, x,
            model.input_indices, model.unit_mask,
            model.activation_indices, model.output_mask,
            model.activation_fns,
            model.input_dim, model.max_layers,
            model.max_units_per_layer, model.buffer_size,
        )
        return jnp.sum(output ** 2)

    _, grads_explicit = eqx.filter_value_and_grad(loss_explicit)(net)
    _, grads_plain = eqx.filter_value_and_grad(loss_plain)(net)

    np.testing.assert_allclose(grads_explicit.weights, grads_plain.weights, atol=1e-5)
    np.testing.assert_allclose(
        grads_explicit.output_weights, grads_plain.output_weights, atol=1e-5
    )
    print("PASS: test_explicit_vjp_matches_autodiff")


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')

    test_shapes()
    test_linear_model_forward()
    test_single_hidden_unit()
    test_tanh_activation()
    test_gradient_only_for_weights()
    test_vmap_batched_forward()
    test_multi_layer_buffer_propagation()
    test_count_helpers()
    test_jit_compilation()
    test_build_outgoing_indices()
    test_sync_outgoing_weights()
    test_custom_vjp_matches_autodiff()
    test_gradient_finite_difference()
    test_gradient_multi_layer()
    test_gradient_multiple_activations()
    test_vmap_grad()
    test_explicit_vjp_matches_optimized()
    test_explicit_vjp_matches_autodiff()

    print("\nAll tests passed!")
