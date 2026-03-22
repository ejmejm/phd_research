"""Tests for ConnectivityManager."""

import jax
import jax.numpy as jnp
import numpy as np

from phd.structure_search.dynamic_network import (
    DynamicNetwork,
    build_outgoing_indices,
    sync_outgoing_weights,
    init_random_dynamic_network,
    count_active_connections,
    count_active_units,
)
from phd.structure_search.connectivity_manager import (
    ConnectivityManager,
    contribution_utility,
    random_generate,
    median_utility_init,
    _unit_buf_positions,
    _reset_optimizer_state,
)
from phd.jax_core.optimizers import EqxOptimizer
from phd.feature_search.jax_core.experiment_helpers import prepare_optimizer
from phd.jax_core.optimizers.adam import custom_optax_adam


def _make_small_network(key=None):
    """Create a small DynamicNetwork with known structure for testing."""
    if key is None:
        key = jax.random.key(42)
    return init_random_dynamic_network(
        input_dim=8,
        output_dim=3,
        n_layers=2,
        units_per_layer=4,
        max_units_per_layer=6,
        max_connections_per_unit=5,
        activations=('relu',),
        connect_all_to_output=True,
        key=key,
    )


def _make_optimizer(model):
    """Create a simple SGD optimizer for a DynamicNetwork."""
    import equinox as eqx
    spec = jax.tree.map(lambda _: False, model)
    spec = eqx.tree_at(lambda n: (n.weights, n.output_weights), spec, (True, True))
    return prepare_optimizer(model, 'sgd', {'learning_rate': 0.01}, filter_spec=spec)


def test_init():
    """ConnectivityManager should initialize with correct shapes and zero stats."""
    net = _make_small_network()
    cm = ConnectivityManager(
        model=net, prune_rate=0.01, connection_budget=100.0,
        decay_rate=0.99, rng=jax.random.key(0),
    )
    assert cm.unit_stats.age.shape == (2, 6)
    assert cm.unit_stats.utility.shape == (2, 6)
    assert float(cm.unit_stats.accumulator) == 0.0
    assert jnp.all(cm.unit_stats.age == 0)
    assert jnp.all(cm.unit_stats.utility == 0)
    print("PASS: test_init")


def test_contribution_utility_shape():
    """Contribution utility should return correct shape and zero for inactive units."""
    net = _make_small_network()
    net = sync_outgoing_weights(net)
    batch_size = 16
    x = jax.random.normal(jax.random.key(1), (batch_size, 8))
    _, buffer = jax.vmap(net)(x)

    utility = contribution_utility(net, buffer)
    assert utility.shape == (2, 6)

    # Inactive unit slots should have zero utility
    inactive = net.unit_mask == 0
    assert jnp.all(utility[inactive] == 0.0)
    print("PASS: test_contribution_utility_shape")


def test_update_stats():
    """update_stats should increment age, update utility, and accumulate budget."""
    net = _make_small_network()
    net = sync_outgoing_weights(net)
    cm = ConnectivityManager(
        model=net, prune_rate=0.01, connection_budget=100.0,
        decay_rate=0.99, rng=jax.random.key(0),
    )

    batch_size = 16
    x = jax.random.normal(jax.random.key(1), (batch_size, 8))
    _, buffer = jax.vmap(net)(x)

    cm2 = cm.update_stats(net, buffer)

    # Age should be 1 for active units, 0 for inactive
    active_mask = net.unit_mask == 1
    assert jnp.all(cm2.unit_stats.age[active_mask] == 1)
    assert jnp.all(cm2.unit_stats.age[~active_mask] == 0)

    # Budget should have increased
    assert float(cm2.unit_stats.accumulator) > 0

    # Utility should be non-negative
    assert jnp.all(cm2.unit_stats.utility >= 0)

    # Second update: age should be 2
    cm3 = cm2.update_stats(net, buffer)
    assert jnp.all(cm3.unit_stats.age[active_mask] == 2)
    assert float(cm3.unit_stats.accumulator) > float(cm2.unit_stats.accumulator)
    print("PASS: test_update_stats")


def test_prune_mask():
    """_make_prune_mask should select lowest-utility active units."""
    net = _make_small_network()
    net = sync_outgoing_weights(net)
    cm = ConnectivityManager(
        model=net, prune_rate=0.5, connection_budget=200.0,
        decay_rate=0.99, maturity_threshold=-1,
        rng=jax.random.key(0),
    )

    # Give some units high utility, some low
    n_active = int(jnp.sum(net.unit_mask))
    utility = jnp.where(
        net.unit_mask == 1,
        jax.random.uniform(jax.random.key(5), (2, 6)),
        0.0,
    )
    # Set a large budget so pruning isn't budget-limited
    from phd.jax_core.utils import tree_replace
    from phd.structure_search.connectivity_manager import UnitStats
    cm = tree_replace(cm, unit_stats=UnitStats(
        age=jnp.ones((2, 6), dtype=jnp.int32) * 10,
        utility=utility,
        accumulator=jnp.array(10000.0),
    ))

    prune_mask, n_pruned = cm._make_prune_mask(net, rng=jax.random.key(1))

    # Should only prune active units
    assert jnp.all(prune_mask <= (net.unit_mask == 1))
    # Should prune some units (budget is large)
    assert int(n_pruned) > 0
    print("PASS: test_prune_mask")


def test_modify_structure_runs():
    """modify_structure should run without errors and maintain valid state."""
    net = _make_small_network()
    net = sync_outgoing_weights(net)
    optimizer = _make_optimizer(net)

    cm = ConnectivityManager(
        model=net, prune_rate=0.5, connection_budget=200.0,
        decay_rate=0.99, maturity_threshold=-1,
        max_new_units_per_step=8,
        rng=jax.random.key(0),
    )

    # Run a few update_stats to build up budget and utility
    batch_size = 16
    x = jax.random.normal(jax.random.key(1), (batch_size, 8))
    _, buffer = jax.vmap(net)(x)
    for i in range(50):
        cm = cm.update_stats(net, buffer)

    # Now modify structure
    cm2, net2, opt2, _, _ = cm.modify_structure(net, optimizer, rng=jax.random.key(2))

    # Basic validity checks
    assert net2.unit_mask.shape == net.unit_mask.shape
    assert net2.weights.shape == net.weights.shape

    # All active units should have valid connections
    for l in range(net2.max_layers):
        for u in range(net2.max_units_per_layer):
            if net2.unit_mask[l, u] == 1:
                # Should have at least one valid connection
                n_valid = int(jnp.sum(net2.input_indices[l, u] >= 0))
                assert n_valid > 0, f"Active unit ({l}, {u}) has no connections"

    print("PASS: test_modify_structure_runs")


def test_connection_budget_invariant():
    """active_connections + budget should remain constant across modify_structure calls."""
    net = _make_small_network()
    net = sync_outgoing_weights(net)
    optimizer = _make_optimizer(net)

    cm = ConnectivityManager(
        model=net, prune_rate=0.1, connection_budget=200.0,
        decay_rate=0.99, maturity_threshold=-1,
        max_new_units_per_step=8,
        rng=jax.random.key(0),
    )

    batch_size = 16
    x = jax.random.normal(jax.random.key(1), (batch_size, 8))
    _, buffer = jax.vmap(net)(x)

    # Build up budget
    for i in range(100):
        cm = cm.update_stats(net, buffer)

    # Record initial total: active_connections + budget
    initial_active = count_active_connections(net)
    initial_budget = float(cm.unit_stats.accumulator)
    initial_total = initial_active + initial_budget

    # Run several modify_structure cycles
    for i in range(5):
        cm, net, optimizer, _, _ = cm.modify_structure(
            net, optimizer, rng=jax.random.key(10 + i),
        )
        net = sync_outgoing_weights(net)

        active = count_active_connections(net)
        budget = float(cm.unit_stats.accumulator)
        total = active + budget

        # Allow small floating point tolerance
        assert abs(total - initial_total) < 1.0, (
            f"Budget invariant violated at step {i}: "
            f"initial_total={initial_total:.1f}, current_total={total:.1f} "
            f"(active={active}, budget={budget:.1f})"
        )

    print("PASS: test_connection_budget_invariant")


def test_generated_units_have_valid_layer_connections():
    """Generated units should only connect to sources in lower layers."""
    net = _make_small_network()
    net = sync_outgoing_weights(net)
    optimizer = _make_optimizer(net)

    cm = ConnectivityManager(
        model=net, prune_rate=0.5, connection_budget=200.0,
        decay_rate=0.99, maturity_threshold=-1,
        max_new_units_per_step=8,
        rng=jax.random.key(0),
    )

    # Build large budget to force lots of generation
    batch_size = 16
    x = jax.random.normal(jax.random.key(1), (batch_size, 8))
    _, buffer = jax.vmap(net)(x)
    for i in range(200):
        cm = cm.update_stats(net, buffer)

    cm2, net2, opt2, _, _ = cm.modify_structure(net, optimizer, rng=jax.random.key(2))

    input_dim = net2.input_dim
    max_units = net2.max_units_per_layer

    for l in range(net2.max_layers):
        for u in range(max_units):
            if net2.unit_mask[l, u] == 1:
                indices = net2.input_indices[l, u]
                valid = indices[indices >= 0]
                for idx in valid:
                    idx_val = int(idx)
                    if idx_val < input_dim:
                        continue  # input positions are always valid
                    # Hidden unit position: determine its layer
                    source_layer = (idx_val - input_dim) // max_units
                    assert source_layer < l, (
                        f"Unit ({l}, {u}) connects to source in layer {source_layer} "
                        f"which is not strictly before layer {l}"
                    )

    print("PASS: test_generated_units_have_valid_layer_connections")


def test_output_connections_for_generated_units():
    """Generated units should have output connections to all output dims."""
    net = _make_small_network()
    net = sync_outgoing_weights(net)
    optimizer = _make_optimizer(net)

    cm = ConnectivityManager(
        model=net, prune_rate=0.5, connection_budget=200.0,
        decay_rate=0.99, maturity_threshold=-1,
        max_new_units_per_step=8,
        rng=jax.random.key(0),
    )

    batch_size = 16
    x = jax.random.normal(jax.random.key(1), (batch_size, 8))
    _, buffer = jax.vmap(net)(x)
    for i in range(200):
        cm = cm.update_stats(net, buffer)

    cm2, net2, opt2, _, _ = cm.modify_structure(net, optimizer, rng=jax.random.key(2))

    # Check all active units have output connections
    buf_positions = _unit_buf_positions(net2)
    for l in range(net2.max_layers):
        for u in range(net2.max_units_per_layer):
            if net2.unit_mask[l, u] == 1:
                bp = int(buf_positions[l, u])
                om = net2.output_mask[:, bp]
                assert jnp.all(om == 1), (
                    f"Active unit ({l}, {u}) at buf_pos {bp} missing output connections"
                )

    print("PASS: test_output_connections_for_generated_units")


def test_optimizer_state_dtype_after_reset():
    """Adam step counts should remain int32 after _reset_optimizer_state."""
    import equinox as eqx
    net = _make_small_network()
    net = sync_outgoing_weights(net)

    # Create Adam optimizer
    spec = jax.tree.map(lambda _: False, net)
    spec = eqx.tree_at(lambda n: (n.weights, n.output_weights), spec, (True, True))
    optimizer = prepare_optimizer(net, 'adam', {'learning_rate': 0.001}, filter_spec=spec)

    # Create an affected mask (pretend we pruned some units)
    affected = jnp.zeros((net.max_layers, net.max_units_per_layer), dtype=jnp.bool_)
    affected = affected.at[0, 0].set(True)

    new_opt = _reset_optimizer_state(optimizer, net, affected)

    # Check that step field dtypes are preserved
    # AdamState is a NamedTuple: (lr, step, exp_avg, exp_avg_sq)
    state = new_opt.state
    if type(state) is tuple:
        state = state[0]

    assert state.step.weights.dtype == jnp.int32, (
        f"Step weights dtype should be int32, got {state.step.weights.dtype}"
    )
    assert state.step.output_weights.dtype == jnp.int32, (
        f"Step output_weights dtype should be int32, got {state.step.output_weights.dtype}"
    )

    # exp_avg should remain float32
    assert state.exp_avg.weights.dtype == jnp.float32
    print("PASS: test_optimizer_state_dtype_after_reset")


def test_empty_init_strategy():
    """DynamicNetwork with init_strategy='empty' should have no connections."""
    net = DynamicNetwork(
        input_dim=8,
        output_dim=3,
        max_layers=2,
        max_units_per_layer=6,
        max_connections_per_unit=5,
        init_strategy='empty',
        key=jax.random.key(42),
    )
    assert jnp.all(net.output_mask == 0), "Empty init should have no output connections"
    assert jnp.all(net.output_weights == 0), "Empty init should have zero output weights"
    assert jnp.all(net.unit_mask == 0), "Empty init should have no active units"

    # Forward pass should produce zeros
    x = jax.random.normal(jax.random.key(1), (8,))
    output, _ = net(x)
    assert jnp.allclose(output, 0.0), "Empty network should output zeros"
    print("PASS: test_empty_init_strategy")


def test_sparse_outgoing_connections():
    """random_sparse strategy should create sparse output + hidden connections."""
    net = init_random_dynamic_network(
        input_dim=8,
        output_dim=3,
        n_layers=2,
        units_per_layer=4,
        max_units_per_layer=6,
        max_connections_per_unit=5,
        max_fan_out=8,
        activations=('relu',),
        connect_all_to_output=True,
        key=jax.random.key(42),
    )
    net = sync_outgoing_weights(net)
    optimizer = _make_optimizer(net)

    cm = ConnectivityManager(
        model=net, prune_rate=0.5, connection_budget=200.0,
        decay_rate=0.99, maturity_threshold=-1,
        max_new_units_per_step=8,
        output_connect_strategy='random_sparse',
        rng=jax.random.key(0),
    )

    batch_size = 16
    x = jax.random.normal(jax.random.key(1), (batch_size, 8))
    _, buffer = jax.vmap(net)(x)
    for i in range(200):
        cm = cm.update_stats(net, buffer)

    cm2, net2, opt2, _, _ = cm.modify_structure(net, optimizer, rng=jax.random.key(2))

    # Check that newly generated units have sparse (not all) output connections
    buf_positions = _unit_buf_positions(net2)
    max_out = max(1, net2.max_fan_out // 2)

    max_out = max(1, net2.max_fan_out // 2)
    for l in range(net2.max_layers):
        for u in range(net2.max_units_per_layer):
            if net2.unit_mask[l, u] == 1:
                bp = int(buf_positions[l, u])
                n_output_conns = int(net2.output_mask[:, bp].sum())
                # Count hidden outgoing: how many later-layer units point to this bp
                n_hidden_out = 0
                for l2 in range(l + 1, net2.max_layers):
                    for u2 in range(net2.max_units_per_layer):
                        if net2.unit_mask[l2, u2] == 1:
                            if jnp.any(net2.input_indices[l2, u2] == bp):
                                n_hidden_out += 1
                total_out = n_output_conns + n_hidden_out
                # Each generated unit should have exactly max_fan_out//2 outgoing
                # (some may be less if not enough targets were available)
                assert total_out >= 1, (
                    f"Unit ({l}, {u}) has zero outgoing connections"
                )
                assert total_out <= max_out, (
                    f"Unit ({l}, {u}) has {total_out} outgoing, max is {max_out}"
                )

    # No hidden unit should exceed max_connections_per_unit
    for l in range(net2.max_layers):
        for u in range(net2.max_units_per_layer):
            if net2.unit_mask[l, u] == 1:
                n_conns = int(jnp.sum(net2.input_indices[l, u] >= 0))
                assert n_conns <= net2.max_connections_per_unit, (
                    f"Unit ({l}, {u}) has {n_conns} connections, "
                    f"max is {net2.max_connections_per_unit}"
                )

    print("PASS: test_sparse_outgoing_connections")


if __name__ == '__main__':
    test_init()
    test_contribution_utility_shape()
    test_update_stats()
    test_prune_mask()
    test_modify_structure_runs()
    test_connection_budget_invariant()
    test_generated_units_have_valid_layer_connections()
    test_output_connections_for_generated_units()
    test_optimizer_state_dtype_after_reset()
    test_empty_init_strategy()
    test_sparse_outgoing_connections()
    print("\nAll tests passed!")
