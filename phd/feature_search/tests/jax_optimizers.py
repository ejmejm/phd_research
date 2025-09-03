import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import DictConfig

from phd.feature_search.jax_core.models import MLP
from phd.feature_search.jax_core.experiment_helpers import prepare_optimizer
from phd.feature_search.jax_core.optimizer import EqxOptimizer


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
        for _ in range(100):
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



if __name__ == '__main__':
    pytest.main([__file__])
