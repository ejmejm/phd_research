"""Summation environment from the UPGD paper (arXiv:2302.03281, Section 4.3).

This implements a nonstationary toy regression task where the target is
y = a * sum(x[i] for i in S), where S is a subset of input indices and a
is a scalar multiplier that can change over time.

Non-stationarity comes in two forms:
1. Changing input-set: S shifts to new indices every `change_subset_freq` steps
2. Changing multiplier: a flips sign every `flip_multiplier_freq` steps
"""
from typing import Optional, Tuple

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
        change_subset_freq: Optional[int] = None,
        flip_multiplier_freq: Optional[int] = None,
        input_min: float = -1.0,
        input_max: float = 1.0,
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
        assert subset_size <= n_features, "Subset size must be less than or equal to number of features!"
        
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
        self.subset_indices = jnp.arange(subset_size)
        self.multiplier = jnp.array(initial_multiplier, dtype=jnp.float32)
        self.step = jnp.array(0, dtype=jnp.int32)
        self.rng = key

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
        new_rng, x_key = random.split(self.rng)
        
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
            candidate_subset = (self.subset_indices + self.subset_size) % self.n_features
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
