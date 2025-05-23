import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phd.feature_search.jax_core.models import ltu, MLP


class TestLTU:
    """Tests for the Linear Threshold Unit (LTU) function."""
    
    def test_ltu_jit_compiles(self):
        """Test that LTU function can be JIT compiled without issues."""
        jitted_ltu = jax.jit(ltu)
        
        x = jnp.array([1.0, -1.0, 0.0])
        result = jitted_ltu(x)
        
        # Should not raise any errors and return expected shape
        assert result.shape == x.shape
        assert result.dtype == jnp.float32
    
    def test_ltu_output_negative_input(self):
        """Test LTU output and gradient for negative input (-1)."""
        x = jnp.array(-1.0)
        
        # Test forward pass
        result = ltu(x)
        expected_output = jnp.array(0.0)  # -1 < 0, so output should be 0
        np.testing.assert_array_almost_equal(result, expected_output)
        
        # Test gradient
        grad_fn = jax.grad(ltu)
        gradient = grad_fn(x)
        
        # Gradient should be sigmoid gradient
        expected_grad = jax.grad(jax.nn.sigmoid)(x)
        np.testing.assert_array_almost_equal(gradient, expected_grad)
    
    def test_ltu_output_zero_input(self):
        """Test LTU output and gradient for zero input (0)."""
        x = jnp.array(0.0)
        
        # Test forward pass
        result = ltu(x)
        expected_output = jnp.array(0.0)  # 0 == 0, so output should be 0 (not > 0)
        np.testing.assert_array_almost_equal(result, expected_output)
        
        # Test gradient
        grad_fn = jax.grad(ltu)
        gradient = grad_fn(x)
        
        # Gradient should be sigmoid gradient
        expected_grad = jax.grad(jax.nn.sigmoid)(x)
        np.testing.assert_array_almost_equal(gradient, expected_grad)
    
    def test_ltu_output_positive_input(self):
        """Test LTU output and gradient for positive input (1)."""
        x = jnp.array(1.0)
        
        # Test forward pass
        result = ltu(x)
        expected_output = jnp.array(1.0)  # 1 > 0, so output should be 1
        np.testing.assert_array_almost_equal(result, expected_output)
        
        # Test gradient
        grad_fn = jax.grad(ltu)
        gradient = grad_fn(x)
        
        # Gradient should be sigmoid gradient
        expected_grad = jax.grad(jax.nn.sigmoid)(x)
        np.testing.assert_array_almost_equal(gradient, expected_grad)
    
    def test_ltu_with_threshold(self):
        """Test LTU with a threshold."""
        x = jnp.array([-1.0, 0.0, 1.0])
        threshold = -0.5
        
        result = ltu(x, threshold)
        expected_output = jnp.array([0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected_output)

        grad = jax.grad(lambda x, t: ltu(x, t).sum())(x, threshold)
        expected_grad = jax.grad(lambda x: jax.nn.sigmoid(x - threshold).sum())(x)
        np.testing.assert_array_almost_equal(grad, expected_grad)


class TestMLP:
    """Tests for the Multi-Layer Perceptron (MLP) class."""
    
    def test_mlp_call_jit_compiles(self):
        """Test that MLP __call__ method can be JIT compiled."""
        key = jax.random.PRNGKey(42)
        mlp = MLP(
            input_dim = 10,
            output_dim = 5,
            n_layers = 2,
            hidden_dim = 20,
            weight_init_method = 'kaiming_uniform',
            activation = 'tanh',
            key = key,
        )
        
        # JIT compile the forward pass
        jitted_forward = eqx.filter_jit(mlp)
        
        x = jnp.ones((10,))
        output, param_inputs = jitted_forward(x)
        
        # Should not raise any errors
        assert output.shape == (5,)
        assert len(param_inputs) == 2  # Two layers
    
    def test_mlp_single_layer(self):
        """Test MLP with single layer (n_layers=1)."""
        key = jax.random.PRNGKey(42)
        mlp = MLP(
            input_dim = 5,
            output_dim = 3,
            n_layers = 1,
            hidden_dim = 10, # Should be ignored for single layer
            weight_init_method = 'kaiming_uniform',
            activation = 'relu',
            key = key,
        )
        
        # Check structure
        assert len(mlp.layers) == 1
        assert mlp.layers[0].weight.shape == (3, 5)  # (output_dim, input_dim)
        
        # Test forward pass
        x = jnp.ones((5,))
        output, param_inputs = mlp(x)
        
        assert output.shape == (3,)
        assert len(param_inputs) == 1
        # Input to the only layer should be x
        np.testing.assert_array_equal(param_inputs[0], x)
    
    def test_mlp_two_layers(self):
        """Test MLP with two layers (n_layers=2)."""
        key = jax.random.PRNGKey(42)
        mlp = MLP(
            input_dim = 4,
            output_dim = 2,
            n_layers = 2,
            hidden_dim = 6,
            weight_init_method = 'kaiming_uniform',
            activation = 'sigmoid',
            key = key,
        )
        
        # Check structure
        assert len(mlp.layers) == 2
        assert mlp.layers[0].weight.shape == (6, 4)  # Hidden layer: (hidden_dim, input_dim)
        assert mlp.layers[1].weight.shape == (2, 6)  # Output layer: (output_dim, hidden_dim)
        
        # Test forward pass
        x = jnp.array([1.0, -1.0, 0.5, -0.5])
        output, param_inputs = mlp(x)
        
        assert output.shape == (2,)
        assert len(param_inputs) == 2
        
        # First layer should receive original input
        np.testing.assert_array_equal(param_inputs[0], x)
        
        # Second layer should receive activated output from first layer
        hidden_output = jax.nn.sigmoid(mlp.layers[0](x))
        np.testing.assert_array_almost_equal(
            param_inputs[1], 
            hidden_output,
        )
    
    def test_mlp_binary_weight_init_first_layer_only(self):
        """Test that binary weight initialization only affects the first layer."""
        key = jax.random.PRNGKey(42)
        mlp = MLP(
            input_dim = 3,
            output_dim = 2,
            n_layers = 2,
            hidden_dim = 8,
            weight_init_method = 'binary',
            activation = 'tanh',
            key = key,
        )
        
        # First layer weights should be binary (-1 or 1)
        first_layer_weights = mlp.layers[0].weight
        unique_weights = jnp.unique(first_layer_weights)
        expected_binary_values = jnp.array([-1.0, 1.0])
        
        # Check that all weights are either -1 or 1
        for weight in unique_weights:
            assert weight in expected_binary_values
        
        # Second layer should use default initialization (not binary)
        second_layer_weights = mlp.layers[1].weight
        unique_second_weights = jnp.unique(second_layer_weights)
        
        # Second layer should have more diverse weights (not just -1 and 1)
        assert len(unique_second_weights) > 2
        assert not jnp.allclose(jnp.abs(second_layer_weights), 1.0)
    
    def test_mlp_zeros_weight_init_first_layer_only(self):
        """Test that zeros weight initialization only affects the first layer."""
        key = jax.random.PRNGKey(42)
        mlp = MLP(
            input_dim = 3,
            output_dim = 2,
            n_layers = 2,
            hidden_dim = 8,
            weight_init_method='zeros',
            activation = 'relu',
            key = key,
        )
        
        # First layer weights should be all zeros
        first_layer_weights = mlp.layers[0].weight
        np.testing.assert_array_equal(first_layer_weights, jnp.zeros_like(first_layer_weights))
        
        # Second layer should use default initialization (not zeros)
        second_layer_weights = mlp.layers[1].weight
        
        # Second layer should have non-zero weights
        assert not jnp.allclose(second_layer_weights, 0.0)
        assert jnp.any(second_layer_weights != 0.0)


if __name__ == '__main__':
    pytest.main([__file__]) 