import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import DictConfig

from phd.feature_search.jax_core.models import MLP
from phd.feature_search.jax_core.experiment_helpers import prepare_optimizer
from phd.feature_search.jax_core.optimizers.optimizer import EqxOptimizer
from functools import partial


class TestOptimizers:
    """Tests for SGD, Adam, and RMSprop optimizers with frozen layers."""
    
    def _create_test_model(self, key):
        """Create a 3-layer MLP with 1 frozen layer for testing."""
        return MLP(
            input_dim = 2,
            output_dim = 1,
            n_layers = 3,
            hidden_dim = 4,
            weight_init_method = 'kaiming_uniform',
            activation = 'tanh',
            n_frozen_layers = 1,  # First layer frozen
            key = key,
        )
    
    def _create_simple_dataset(self, key, n_samples = 5):
        """Create a simple dataset: y = x1 + x2."""
        x_key, y_key = jax.random.split(key)
        x = jax.random.normal(x_key, (n_samples, 2))
        y = jnp.sum(x, axis=1, keepdims=True)  # Simple sum function
        return x, y
    
    def _train_step(self, model, optimizer, x_batch, y_batch):
        """Single training step."""
        def loss_fn(model):
            predictions, _ = jax.vmap(model)(x_batch)
            return jnp.mean((predictions - y_batch) ** 2)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        updates, optimizer = optimizer.with_update(grads, model)
        model = eqx.apply_updates(model, updates)
        
        return model, optimizer, loss
    
    def test_sgd_optimizer(self):
        """Test SGD optimizer reduces loss and respects frozen layers."""
        key = jax.random.PRNGKey(42)
        model_key, data_key = jax.random.split(key)
        
        # Create model and data
        model = self._create_test_model(model_key)
        x_data, y_data = self._create_simple_dataset(data_key)
        
        # Store initial weights for comparison
        initial_first_layer_weights = model.layers[0].weight.copy()
        initial_second_layer_weights = model.layers[1].weight.copy()
        initial_third_layer_weights = model.layers[2].weight.copy()
        
        # Create optimizer
        optimizer_config = DictConfig({'learning_rate': 0.03})
        optimizer = prepare_optimizer(model, 'sgd', optimizer_config)
        
        # Initial loss
        def loss_fn(model):
            predictions, _ = jax.vmap(model)(x_data)
            return jnp.mean((predictions - y_data) ** 2)
        
        initial_loss = loss_fn(model)
        
        # Train for several steps
        current_model = model
        current_optimizer = optimizer
        for _ in range(100):
            # print(current_optimizer.optimizer_state)
            current_model, current_optimizer, loss = self._train_step(
                current_model, current_optimizer, x_data, y_data
            )
        
        final_loss = loss_fn(current_model)
        
        # Verify loss reduction
        assert final_loss < initial_loss * 0.2, f"Loss should reduce significantly: {initial_loss} -> {final_loss}"
        
        # Verify first layer is frozen (unchanged)
        np.testing.assert_array_equal(
            current_model.layers[0].weight,
            initial_first_layer_weights,
            "First layer weights should remain unchanged (frozen)"
        )
        
        # Verify other layers have changed
        assert not jnp.allclose(
            current_model.layers[1].weight,
            initial_second_layer_weights,
            atol=1e-6
        ), "Second layer weights should have changed"
        
        assert not jnp.allclose(
            current_model.layers[2].weight,
            initial_third_layer_weights,
            atol=1e-6
        ), "Third layer weights should have changed"
    
    
    def test_sgd_optimizer_with_jit(self):
        """Test SGD optimizer reduces loss and respects frozen layers."""
        key = jax.random.PRNGKey(42)
        model_key, data_key = jax.random.split(key)
        
        # Create model and data
        model = self._create_test_model(model_key)
        x_data, y_data = self._create_simple_dataset(data_key)
        
        # Store initial weights for comparison
        initial_first_layer_weights = model.layers[0].weight.copy()
        initial_second_layer_weights = model.layers[1].weight.copy()
        initial_third_layer_weights = model.layers[2].weight.copy()
        
        # Create optimizer
        optimizer_config = DictConfig({'learning_rate': 0.03})
        optimizer = prepare_optimizer(model, 'sgd', optimizer_config)
        
        # Initial loss
        def loss_fn(model):
            predictions, _ = jax.vmap(model)(x_data)
            return jnp.mean((predictions - y_data) ** 2)
        
        initial_loss = loss_fn(model)
        
        # Train for several steps
        current_model = model
        current_optimizer = optimizer
        train_step = jax.jit(self._train_step)
        for _ in range(100):
            # print(current_optimizer.optimizer_state)
            current_model, current_optimizer, loss = train_step(
                current_model, current_optimizer, x_data, y_data
            )
        
        final_loss = loss_fn(current_model)
        
        # Verify loss reduction
        assert final_loss < initial_loss * 0.2, f"Loss should reduce significantly: {initial_loss} -> {final_loss}"
        
        # Verify first layer is frozen (unchanged)
        np.testing.assert_array_equal(
            current_model.layers[0].weight,
            initial_first_layer_weights,
            "First layer weights should remain unchanged (frozen)"
        )
        
        # Verify other layers have changed
        assert not jnp.allclose(
            current_model.layers[1].weight,
            initial_second_layer_weights,
            atol=1e-6
        ), "Second layer weights should have changed"
        
        assert not jnp.allclose(
            current_model.layers[2].weight,
            initial_third_layer_weights,
            atol=1e-6
        ), "Third layer weights should have changed"
    
    
    def test_adam_optimizer(self):
        """Test SGD optimizer reduces loss and respects frozen layers."""
        key = jax.random.PRNGKey(42)
        model_key, data_key = jax.random.split(key)
        
        # Create model and data
        model = self._create_test_model(model_key)
        x_data, y_data = self._create_simple_dataset(data_key)
        
        # Store initial weights for comparison
        initial_first_layer_weights = model.layers[0].weight.copy()
        initial_second_layer_weights = model.layers[1].weight.copy()
        initial_third_layer_weights = model.layers[2].weight.copy()
        
        # Create optimizer
        optimizer_config = DictConfig({'learning_rate': 0.01})
        optimizer = prepare_optimizer(model, 'adam', optimizer_config)
        
        # Initial loss
        def loss_fn(model):
            predictions, _ = jax.vmap(model)(x_data)
            return jnp.mean((predictions - y_data) ** 2)
        
        initial_loss = loss_fn(model)
        
        # Train for several steps
        current_model = model
        current_optimizer = optimizer
        train_step = jax.jit(self._train_step)
        for _ in range(500):
            current_model, current_optimizer, loss = train_step(
                current_model, current_optimizer, x_data, y_data
            )
        
        final_loss = loss_fn(current_model)
        
        # Verify loss reduction
        assert final_loss < initial_loss * 0.2, f"Loss should reduce significantly: {initial_loss} -> {final_loss}"
        
        # Verify first layer is frozen (unchanged)
        np.testing.assert_array_equal(
            current_model.layers[0].weight,
            initial_first_layer_weights,
            "First layer weights should remain unchanged (frozen)"
        )
        
        # Verify other layers have changed
        assert not jnp.allclose(
            current_model.layers[1].weight,
            initial_second_layer_weights,
            atol=1e-6
        ), "Second layer weights should have changed"
        
        assert not jnp.allclose(
            current_model.layers[2].weight,
            initial_third_layer_weights,
            atol=1e-6
        ), "Third layer weights should have changed"


class TestIDBD:
    """Tests for IDBD optimizer batch size equivalence."""
    
    def _create_simple_model(self, key):
        """Create a simple 2-layer MLP for testing."""
        return MLP(
            input_dim = 3,
            output_dim = 1,
            n_layers = 2,
            hidden_dim = 4,
            weight_init_method = 'kaiming_uniform',
            activation = 'tanh',
            n_frozen_layers = 0,
            key = key,
        )
    
    def _compute_idbd_gradients(self, model, x_batch, y_batch):
        """Compute both loss gradients and prediction gradients for IDBD."""
        def compute_loss(model, inputs, targets):
            outputs, param_inputs = jax.vmap(partial(model, key=None))(inputs)
            loss = jnp.mean((outputs - targets) ** 2)
            return loss, (outputs, param_inputs)
        
        # Loss gradients
        (loss, (outputs, param_inputs)), loss_grads = jax.value_and_grad(
            compute_loss, has_aux=True)(model, x_batch, y_batch)
        
        # Prediction gradients (for IDBD)
        output_grads = jax.grad(
            lambda m, x: jax.vmap(partial(
                m, key=None))(x)[0].mean())(model, x_batch)
        
        return loss_grads, output_grads
    
    @pytest.mark.parametrize("autostep", [True, False])
    def test_idbd_batch_size_equivalence(self, autostep):
        """Test that IDBD updates are equivalent for batch_size=1 (4 steps) vs batch_size=4 (1 step)."""
        key = jax.random.PRNGKey(42)
        model_key, data_key = jax.random.split(key)
        
        # Create model
        model = self._create_simple_model(model_key)
        
        # Create data: 4 samples
        x_key, y_key = jax.random.split(data_key)
        x_all = jax.random.normal(x_key, (4, 3))
        y_all = jnp.sum(x_all, axis=1, keepdims=True)  # Simple sum function
        
        # Create optimizer config
        optimizer_config = DictConfig({
            'learning_rate': 0.01,
            'meta_learning_rate': 0.005,
            'autostep': autostep,
        })
        
        # Store initial parameters (will be reused)
        initial_params = jax.tree.map(lambda x: x.copy(), model)
        
        # Case 1: batch_size=1, 4 steps (updates not applied to parameters)
        model_bs1 = jax.tree.map(lambda x: x.copy(), initial_params)
        optimizer_bs1_initial = prepare_optimizer(model_bs1, 'idbd', optimizer_config)
        initial_state_bs1 = optimizer_bs1_initial.state
        
        param_updates_list = []
        beta_changes_list = []
        h_changes_list = []
        
        for i in range(4):
            x_batch = x_all[i:i+1]
            y_batch = y_all[i:i+1]
            
            loss_grads, output_grads = self._compute_idbd_gradients(
                model_bs1, x_batch, y_batch)
            
            # Get updates and new state using the initial optimizer (not updated)
            updates, updated_optimizer = optimizer_bs1_initial.with_update(
                (loss_grads, output_grads), model_bs1)
            
            param_updates_list.append(updates)
            
            # Track beta and h changes from initial state
            beta_change = jax.tree.map(
                lambda new, old: new - old,
                updated_optimizer.state.beta, initial_state_bs1.beta)
            h_change = jax.tree.map(
                lambda new, old: new - old,
                updated_optimizer.state.h, initial_state_bs1.h)
            
            beta_changes_list.append(beta_change)
            h_changes_list.append(h_change)
        
        # Compute mean state updates
        param_updates_mean_bs1 = jax.tree.map(
            lambda *updates: jnp.stack(updates).mean(axis=0), *param_updates_list)
        beta_changes_mean_bs1 = jax.tree.map(
            lambda *updates: jnp.stack(updates).mean(axis=0), *beta_changes_list)
        h_changes_mean_bs1 = jax.tree.map(
            lambda *updates: jnp.stack(updates).mean(axis=0), *h_changes_list)
        
        # Case 2: batch_size=4, 1 step
        model_bs4 = jax.tree.map(lambda x: x.copy(), initial_params)
        optimizer_bs4 = prepare_optimizer(model_bs4, 'idbd', optimizer_config)
        initial_state_bs4 = optimizer_bs4.state
        
        x_batch = x_all
        y_batch = y_all
        
        loss_grads, output_grads = self._compute_idbd_gradients(
            model_bs4, x_batch, y_batch)
        
        param_updates_bs4, optimizer_bs4 = optimizer_bs4.with_update(
            (loss_grads, output_grads), model_bs4)
        
        final_state_bs4 = optimizer_bs4.state
        beta_changes_bs4 = jax.tree.map(
            lambda new, old: new - old,
            final_state_bs4.beta, initial_state_bs4.beta)
        h_changes_bs4 = jax.tree.map(
            lambda new, old: new - old,
            final_state_bs4.h, initial_state_bs4.h)
        
        param_updates_mean_bs4 = param_updates_bs4
        beta_changes_mean_bs4 = beta_changes_bs4
        h_changes_mean_bs4 = h_changes_bs4

        def compare_updates(upd1, upd2, name, atol=1e-5, rtol=1e-5):
            """Compare two update or state change trees."""
            def _compare(a, b):
                if not jnp.allclose(a, b, atol=atol, rtol=rtol):
                    max_diff = jnp.abs(a - b).max()
                    mean_diff = jnp.abs(a - b).mean()
                    raise AssertionError(
                        f"{name} differ: max_diff={max_diff}, "
                        f"mean_diff={mean_diff}, shape={a.shape}")
                return True
            jax.tree.map(_compare, upd1, upd2)

        compare_updates(param_updates_mean_bs1, param_updates_mean_bs4, "Parameter", atol=1e-5, rtol=1e-5)
        compare_updates(beta_changes_mean_bs1, beta_changes_mean_bs4, "Beta", atol=1e-4, rtol=1e-4)
        compare_updates(h_changes_mean_bs1, h_changes_mean_bs4, "H", atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__])
