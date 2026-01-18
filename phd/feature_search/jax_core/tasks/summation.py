"""Summation environment from the UPGD paper (arXiv:2302.03281, Section 4.3).

This implements a nonstationary toy regression task where the target is
y = a * sum(x[i] for i in S), where S is a subset of input indices and a
is a scalar multiplier that can change over time.

Non-stationarity comes in two forms:
1. Changing input-set: S shifts to new indices every `change_subset_freq` steps
2. Changing multiplier: a flips sign every `flip_multiplier_freq` steps
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int

from ..utils import tree_replace


class SummationTask(eqx.Module):
    """Nonstationary summation regression task from UPGD paper.
    
    The target function is y = a * sum(x[i] for i in S), where:
    - x is a random input vector sampled uniformly from [input_min, input_max]
    - S is a subset of indices that can change periodically
    - a is a scalar multiplier that can flip sign periodically
    
    This is a simple regression environment for testing continual learning
    algorithms under distribution shift.
    """
    
    # Static parameters (configuration)
    n_features: int = eqx.field(static=True)
    subset_size: int = eqx.field(static=True)
    change_subset_freq: Optional[int] = eqx.field(static=True)
    flip_multiplier_freq: Optional[int] = eqx.field(static=True)
    input_min: float = eqx.field(static=True)
    input_max: float = eqx.field(static=True)
    initial_multiplier: float = eqx.field(static=True)
    
    # Dynamic parameters (state)
    subset_indices: Int[Array, 'subset_size']
    multiplier: Float[Array, '']
    step: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_features: int = 16,
        subset_size: int = 2,
        change_subset_freq: Optional[int] = 200,
        flip_multiplier_freq: Optional[int] = 200,
        input_min: float = -0.5,
        input_max: float = 0.5,
        initial_multiplier: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize the summation task.
        
        Args:
            n_features: Number of input features (default 16 per paper).
            subset_size: Number of indices in subset S (default 2 per paper).
            change_subset_freq: Steps between subset changes. None to never change.
            flip_multiplier_freq: Steps between multiplier sign flips. None to
                never flip.
            input_min: Minimum value for uniform input sampling (default -0.5).
            input_max: Maximum value for uniform input sampling (default 0.5).
            initial_multiplier: Initial value of the multiplier a (default 1.0).
            seed: Random seed for reproducibility.
        """
        # Store static configuration
        self.n_features = n_features
        self.subset_size = subset_size
        self.change_subset_freq = change_subset_freq
        self.flip_multiplier_freq = flip_multiplier_freq
        self.input_min = input_min
        self.input_max = input_max
        self.initial_multiplier = initial_multiplier
        
        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)
        
        # Initialize dynamic state
        key, subset_key = random.split(key)
        self.subset_indices = random.choice(
            subset_key,
            n_features,
            shape = (subset_size,),
            replace = False,
        )
        self.multiplier = jnp.array(initial_multiplier, dtype=jnp.float32)
        self.step = jnp.array(0, dtype=jnp.int32)
        self.rng = key

    def _sample_new_subset(
        self, key: random.PRNGKey,
    ) -> Int[Array, 'subset_size']:
        """Sample a new random subset of indices.
        
        Args:
            key: PRNG key for random sampling.
            
        Returns:
            Array of randomly selected indices.
        """
        return random.choice(
            key,
            self.n_features,
            shape = (self.subset_size,),
            replace = False,
        )

    def _compute_target(
        self, x: Float[Array, 'batch_size n_features'],
    ) -> Float[Array, 'batch_size 1']:
        """Compute target y = a * sum(x[i] for i in S).
        
        Args:
            x: Input features of shape (batch_size, n_features).
            
        Returns:
            Target values of shape (batch_size, 1).
        """
        # Select features at subset indices and sum them
        x_subset = x[:, self.subset_indices]
        y = self.multiplier * jnp.sum(x_subset, axis=1, keepdims=True)
        return y

    def generate_batch(
        self, batch_size: int = 1,
    ) -> Tuple[eqx.Module, Tuple[Float[Array, 'batch n_features'], Float[Array, 'batch 1']]]:
        """Generate a batch of (input, target) pairs and update task state.
        
        The task state is updated after generating the batch:
        - Step counter increments
        - Subset may change if change_subset_freq steps have passed
        - Multiplier may flip if flip_multiplier_freq steps have passed
        
        Args:
            batch_size: Number of samples to generate.
            
        Returns:
            Tuple containing:
            - New task state with updated step/subset/multiplier
            - Batch data (x, y) where x is inputs and y is targets
        """
        new_rng, x_key, subset_key = random.split(self.rng, 3)
        
        # Sample random inputs uniformly
        x = random.uniform(
            x_key,
            shape = (batch_size, self.n_features),
            minval = self.input_min,
            maxval = self.input_max,
        )
        
        # Compute targets with current subset and multiplier
        y = self._compute_target(x)
        
        # Increment step counter
        new_step = self.step + 1
        
        # Maybe update subset indices
        new_subset = self.subset_indices
        if self.change_subset_freq is not None:
            should_change_subset = (new_step % self.change_subset_freq) == 0
            candidate_subset = self._sample_new_subset(subset_key)
            new_subset = jnp.where(
                should_change_subset,
                candidate_subset,
                self.subset_indices,
            )
        
        # Maybe flip multiplier sign
        new_multiplier = self.multiplier
        if self.flip_multiplier_freq is not None:
            should_flip = (new_step % self.flip_multiplier_freq) == 0
            new_multiplier = jnp.where(
                should_flip,
                -self.multiplier,
                self.multiplier,
            )
        
        # Create new task state
        new_task_state = tree_replace(
            self,
            subset_indices = new_subset,
            multiplier = new_multiplier,
            step = new_step,
            rng = new_rng,
        )
        
        return new_task_state, (x, y)


class SummationTaskWithSubspaceInputs(eqx.Module):
    """Summation task with shifting input subspace (similar to InputChangingGEOFFTask).
    
    Like SummationTask, but inputs are sampled from a small subspace that shifts
    periodically within the full input bounds. This creates additional 
    non-stationarity in the input distribution.
    """
    
    # Static parameters (configuration)
    n_features: int = eqx.field(static=True)
    subset_size: int = eqx.field(static=True)
    change_subset_freq: Optional[int] = eqx.field(static=True)
    flip_multiplier_freq: Optional[int] = eqx.field(static=True)
    input_bounds: Tuple[float, float] = eqx.field(static=True)
    input_subspace_range: float = eqx.field(static=True)
    input_change_freq: Optional[int] = eqx.field(static=True)
    max_input_center_change: float = eqx.field(static=True)
    initial_multiplier: float = eqx.field(static=True)
    
    # Dynamic parameters (state)
    subset_indices: Int[Array, 'subset_size']
    multiplier: Float[Array, '']
    input_subspace_centers: Float[Array, 'n_features']
    step: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_features: int = 16,
        subset_size: int = 2,
        change_subset_freq: Optional[int] = 200,
        flip_multiplier_freq: Optional[int] = 200,
        input_bounds: Tuple[float, float] = (-0.5, 0.5),
        input_subspace_range: float = 0.1,
        input_change_freq: Optional[int] = None,
        max_input_center_change: float = 0.1,
        initial_multiplier: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize the summation task with subspace inputs.
        
        Args:
            n_features: Number of input features.
            subset_size: Number of indices in subset S.
            change_subset_freq: Steps between subset changes. None to never change.
            flip_multiplier_freq: Steps between multiplier sign flips. None to
                never flip.
            input_bounds: Overall bounds of the input space.
            input_subspace_range: Range of the uniform distribution for sampling
                inputs around the subspace center.
            input_change_freq: Steps between input subspace center shifts. None
                to never change.
            max_input_center_change: Maximum shift of subspace center per change.
            initial_multiplier: Initial value of the multiplier a.
            seed: Random seed for reproducibility.
        """
        # Store static configuration
        self.n_features = n_features
        self.subset_size = subset_size
        self.change_subset_freq = change_subset_freq
        self.flip_multiplier_freq = flip_multiplier_freq
        self.input_bounds = input_bounds
        self.input_subspace_range = input_subspace_range
        self.input_change_freq = input_change_freq
        self.max_input_center_change = max_input_center_change
        self.initial_multiplier = initial_multiplier
        
        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)
        
        # Initialize subset indices
        key, subset_key = random.split(key)
        self.subset_indices = random.choice(
            subset_key,
            n_features,
            shape = (subset_size,),
            replace = False,
        )
        
        # Initialize input subspace centers
        key, center_key = random.split(key)
        self.input_subspace_centers = random.uniform(
            center_key,
            shape = (n_features,),
            minval = input_bounds[0],
            maxval = input_bounds[1],
        )
        
        self.multiplier = jnp.array(initial_multiplier, dtype=jnp.float32)
        self.step = jnp.array(0, dtype=jnp.int32)
        self.rng = key

    def _sample_new_subset(
        self, key: random.PRNGKey,
    ) -> Int[Array, 'subset_size']:
        """Sample a new random subset of indices."""
        return random.choice(
            key,
            self.n_features,
            shape = (self.subset_size,),
            replace = False,
        )

    def _compute_updated_input_subspace_centers(
        self, key: random.PRNGKey,
    ) -> Float[Array, 'n_features']:
        """Compute new input subspace centers with random shift."""
        center_shifts = random.uniform(
            key,
            shape = (self.n_features,),
            minval = -self.max_input_center_change,
            maxval = self.max_input_center_change,
        )
        new_centers = self.input_subspace_centers + center_shifts
        # Wrap around within bounds
        min_bound, max_bound = self.input_bounds
        range_size = max_bound - min_bound
        new_centers = min_bound + jnp.mod(new_centers - min_bound, range_size)
        return new_centers

    def _sample_inputs(
        self, key: random.PRNGKey, batch_size: int = 1,
    ) -> Float[Array, 'batch_size n_features']:
        """Sample inputs from current subspace."""
        bound = self.input_subspace_range / 2.0
        inputs = random.uniform(
            key,
            shape = (batch_size, self.n_features),
            minval = -bound,
            maxval = bound,
        )
        inputs = inputs + jnp.expand_dims(self.input_subspace_centers, 0)
        # Wrap around within bounds
        min_val, max_val = self.input_bounds
        inputs = min_val + jnp.mod(inputs - min_val, max_val - min_val)
        return inputs

    def _compute_target(
        self, x: Float[Array, 'batch_size n_features'],
    ) -> Float[Array, 'batch_size 1']:
        """Compute target y = a * sum(x[i] for i in S)."""
        x_subset = x[:, self.subset_indices]
        y = self.multiplier * jnp.sum(x_subset, axis=1, keepdims=True)
        return y

    def generate_batch(
        self, batch_size: int = 1,
    ) -> Tuple[eqx.Module, Tuple[Float[Array, 'batch n_features'], Float[Array, 'batch 1']]]:
        """Generate a batch of (input, target) pairs and update task state.
        
        Args:
            batch_size: Number of samples to generate.
            
        Returns:
            Tuple containing:
            - New task state with updated step/subset/multiplier/centers
            - Batch data (x, y) where x is inputs and y is targets
        """
        new_rng, x_key, subset_key, center_key = random.split(self.rng, 4)
        
        # Sample inputs from current subspace
        x = self._sample_inputs(x_key, batch_size)
        
        # Compute targets
        y = self._compute_target(x)
        
        # Increment step counter
        new_step = self.step + 1
        
        # Maybe update subset indices
        new_subset = self.subset_indices
        if self.change_subset_freq is not None:
            should_change_subset = (new_step % self.change_subset_freq) == 0
            candidate_subset = self._sample_new_subset(subset_key)
            new_subset = jnp.where(
                should_change_subset,
                candidate_subset,
                self.subset_indices,
            )
        
        # Maybe flip multiplier sign
        new_multiplier = self.multiplier
        if self.flip_multiplier_freq is not None:
            should_flip = (new_step % self.flip_multiplier_freq) == 0
            new_multiplier = jnp.where(
                should_flip,
                -self.multiplier,
                self.multiplier,
            )
        
        # Maybe update input subspace centers
        new_centers = self.input_subspace_centers
        if self.input_change_freq is not None:
            should_change_centers = (new_step % self.input_change_freq) == 0
            candidate_centers = self._compute_updated_input_subspace_centers(center_key)
            new_centers = jnp.where(
                should_change_centers,
                candidate_centers,
                self.input_subspace_centers,
            )
        
        # Create new task state
        new_task_state = tree_replace(
            self,
            subset_indices = new_subset,
            multiplier = new_multiplier,
            input_subspace_centers = new_centers,
            step = new_step,
            rng = new_rng,
        )
        
        return new_task_state, (x, y)
