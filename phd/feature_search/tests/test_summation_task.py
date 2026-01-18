"""Tests for the SummationTask environment from the UPGD paper."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phd.feature_search.jax_core.tasks.summation import SummationTask


class TestSummationTaskStationary:
    """Tests for SummationTask with no changes (stationary setting)."""
    
    def test_target_is_sum_of_all_features(self):
        """Test that target y = multiplier * sum(x) when using all features."""
        task = SummationTask(
            n_features = 4,
            subset_size = 2,
            change_subset_freq = None,
            flip_multiplier_freq = None,
            initial_multiplier = 1.0,
            seed = 42,
        )
        
        # Generate a batch
        new_task, (x, y) = task.generate_batch(batch_size=2)
        
        # Manually compute expected target: y = 1.0 * sum(x)
        expected_y = jnp.sum(x[:, :2], axis=1, keepdims=True)
        
        np.testing.assert_array_almost_equal(y, expected_y)
    
    def test_multiplier_applied_correctly(self):
        """Test that the multiplier scales the sum correctly."""
        task = SummationTask(
            n_features = 4,
            subset_size = 2,
            change_subset_freq = None,
            flip_multiplier_freq = None,
            initial_multiplier = 2.5,
            seed = 42,
        )
        
        new_task, (x, y) = task.generate_batch(batch_size=5)
        
        # Expected: y = 2.5 * sum(x)
        expected_y = 2.5 * jnp.sum(x[:, :2], axis=1, keepdims=True)
        
        np.testing.assert_array_almost_equal(y, expected_y)
    
    def test_step_increments_without_changes(self):
        """Test that step counter increments even without any changes."""
        task = SummationTask(
            n_features = 4,
            subset_size = 2,
            change_subset_freq = None,
            flip_multiplier_freq = None,
            seed = 42,
        )
        
        assert task.step == 0
        
        task, _ = task.generate_batch()
        assert task.step == 1
        
        task, _ = task.generate_batch()
        assert task.step == 2


class TestSummationTaskMultiplierFlipping:
    """Tests for SummationTask with multiplier flipping."""
    
    def test_multiplier_flips_at_correct_frequency(self):
        """Test that multiplier flips sign at the specified frequency."""
        task = SummationTask(
            n_features = 4,
            subset_size = 2,
            change_subset_freq = None,
            flip_multiplier_freq = 4,
            initial_multiplier = 1.0,
            seed = 42,
        )
        
        # Step 0 -> 1, 2, 3: multiplier should be 1.0
        # Step 4: multiplier flips to -1.0
        # Step 5, 6, 7: multiplier should be -1.0
        # Step 8: multiplier flips back to 1.0
        
        expected_multipliers = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0]
        
        for i, expected in enumerate(expected_multipliers):
            task, (x, y) = task.generate_batch(batch_size=5)
            expected_y = expected * jnp.sum(x[:, :2], axis=1, keepdims=True)
            np.testing.assert_array_almost_equal(y, expected_y)


class TestSummationTaskSubsetChange:
    """Tests for SummationTask with subset changing."""
    
    def test_subset_indices_shift_at_correct_frequency(self):
        """Test that subset indices shift at the specified frequency."""
        task = SummationTask(
            n_features = 5,
            subset_size = 2,
            change_subset_freq = 3,
            flip_multiplier_freq = None,
            seed = 42,
        )
        
        exptected_indices = jnp.array([
            [0, 1], [0, 1], [0, 1],
            [2, 3], [2, 3], [2, 3],
            [4, 0], [4, 0], [4, 0],
        ])
        
        for expected in exptected_indices:
            task, (x, y) = task.generate_batch(batch_size=5)
            expected_y = jnp.sum(x[:, expected], axis=1, keepdims=True)
            np.testing.assert_array_almost_equal(y, expected_y)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])