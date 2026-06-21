from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int

from phd.jax_core.utils import tree_replace


class FeatureSiftingTask(eqx.Module):
    """An online feature-selection ("sifting") regression task.

    A fixed target function generates each sample's label as a weighted sum
    of ``n_target_features`` true features drawn uniformly from [-1, 1], with
    weights fixed at +1 or -1 (plus optional noise on the target output).

    The learner never sees the true features directly. Instead it observes
    ``n_learner_features`` candidate features, each of which is a noisy view of
    one randomly chosen target feature: a feature mixes the true value with
    pure noise according to its own noise coefficient, so some candidates are
    clean and informative while others are mostly (or entirely) noise.
    Multiple candidates may also point at the same target feature, and not
    every target feature is guaranteed to be covered.

    The learner's job is to predict the target output from these candidates,
    which requires sifting the useful features from the useless ones. On each
    ``step`` the learner supplies a ``prune_mask`` marking candidates to
    discard; pruned slots are replaced with freshly generated features (a new
    random target index and a new noise coefficient). This makes the task a
    continual generate-and-test / feature-search problem: the learner must
    repeatedly throw away low-utility features and hope their replacements are
    better, while exploiting the good features it has found so far.
    """
    
    # Static parameters (configuration)
    n_target_features: int = eqx.field(static=True)
    n_learner_features: int = eqx.field(static=True)
    flip_period: int = eqx.field(static=True)
    
    # Dynamic parameters (weights and state)
    learner_feature_idxs: Int[Array, 'n_learner_features']
    noise_coefficients: Float[Array, 'n_learner_features']
    weights: List[Float[Array, 'n_target_features']]
    target_noise_std: float
    step_idx: Int[Array, '']
    rng: random.PRNGKey

    def __init__(
        self,
        n_target_features: int = 10,
        n_learner_features: int = 20,
        target_noise_std: float = 0.0,
        flip_period: int = -1,
        key: random.PRNGKey = None,
    ):
        """
        Args:
            n_target_features: Number of features in the target network
            n_learner_features: Number of features in the learner network
            target_noise_std: Standard deviation of the noise added to the target features
            seed: Random seed for reproducibility
        """
        self.n_target_features = n_target_features
        self.n_learner_features = n_learner_features
        self.target_noise_std = target_noise_std
        self.flip_period = flip_period
        self.step_idx = 0
        
        # Set up RNG
        if key is None:
            key = random.PRNGKey(np.random.randint(0, 2**31))
        self.rng, feature_key, weight_key = random.split(key, 3)
        
        # Initialize learner features
        self.learner_feature_idxs, self.noise_coefficients = self._generate_new_features(
            feature_key, self.n_learner_features,
        )

        # Initialize target weights, random -1 or +1
        self.weights = jax.random.bernoulli(weight_key, 0.5, (self.n_target_features,)) * 2 - 1

    def _generate_new_features(
        self, key: random.PRNGKey, n: int,
    ) -> Tuple[Int[Array, 'n'], Float[Array, 'n']]:
        """Generates ``n`` new feature indices and noise coefficients.

        Returns:
            Tuple containing:
            - Array of shape (n,) of target feature indices.
            - Array of shape (n,) of noise coefficients (uniform in [0, 1]).
        """
        idx_key, noise_key = random.split(key)
        feature_idxs = random.randint(idx_key, (n,), 0, self.n_target_features)
        noise_coefficients = random.uniform(noise_key, (n,), minval=0, maxval=1)
        return feature_idxs, noise_coefficients
    
    def step(
        self,
        prune_mask: Int[Array, 'n_learner_features'],
    )-> Tuple['FeatureSiftingTask', Tuple[Float[Array, 'n_learner_features'], Float[Array, '']]]:
        """
        Generate a new sample from the feature sifting task.

        Returns:
            Tuple containing:
            - An updated FeatureSiftingTask state with an updated RNG.
            - (learner_features, target_output):
                - learner_features: Array of shape (n_learner_features,), the noisy features sampled by the learner.
                - target_output: Scalar float, the weighted sum of the true target features.
        """
        new_rng, flip_key, feature_key, x_key, target_noise_key, feature_noise_key = random.split(self.rng, 6)
        
        new_weights = self.weights
        if self.flip_period > 0:
            should_flip = (self.step_idx % self.flip_period) == 0
            flip_idx = random.randint(flip_key, (1,), 0, self.n_target_features)
            new_weights = new_weights.at[flip_idx].set(-new_weights[flip_idx])
            new_weights = jnp.where(should_flip, new_weights, self.weights)

        new_state: 'FeatureSiftingTask' = tree_replace(
            self,
            weights = new_weights,
            step_idx = self.step_idx + 1,
        )

        # Generate new features
        new_feature_idxs, new_noise_coefficients = self._generate_new_features(
            feature_key, self.n_learner_features,
        )
        feature_idxs = jnp.where(prune_mask, new_feature_idxs, self.learner_feature_idxs)
        noise_coefficients = jnp.where(prune_mask, new_noise_coefficients, self.noise_coefficients)
    
        # Update state
        new_state: 'FeatureSiftingTask' = tree_replace(
            self,
            rng = new_rng,
            learner_feature_idxs = feature_idxs,
            noise_coefficients = noise_coefficients,
        ) 
        
        # Generate target features and output
        target_features = jax.random.uniform(x_key, (new_state.n_target_features,), minval=-1, maxval=1)
        target_output = jnp.sum(target_features * new_state.weights)
        target_output += jax.random.normal(target_noise_key, (1,)) * new_state.target_noise_std
        
        # Generate learner features
        learner_features = target_features[new_state.learner_feature_idxs]
        feature_noise = jax.random.uniform(feature_noise_key, (new_state.n_learner_features,), minval=-1, maxval=1)
        learner_features = (1 - new_state.noise_coefficients) * learner_features + new_state.noise_coefficients * feature_noise
        
        return new_state, (learner_features, target_output)


class FeatureSiftingTaskNP:
    """A numpy port of :class:`FeatureSiftingTask` (see ``feature_sifting.py``).

    An online feature-selection ("sifting") regression task.

    A fixed target function generates each sample's label as a weighted sum
    of ``n_target_features`` true features drawn uniformly from [-1, 1], with
    weights fixed at +1 or -1 (plus optional noise on the target output).

    The learner never sees the true features directly. Instead it observes
    ``n_learner_features`` candidate features, each of which is a noisy view of
    one randomly chosen target feature: a feature mixes the true value with
    pure noise according to its own noise coefficient, so some candidates are
    clean and informative while others are mostly (or entirely) noise.
    Multiple candidates may also point at the same target feature, and not
    every target feature is guaranteed to be covered.

    The learner's job is to predict the target output from these candidates,
    which requires sifting the useful features from the useless ones. On each
    ``step`` the learner supplies a ``prune_mask`` marking candidates to
    discard; pruned slots are replaced with freshly generated features (a new
    random target index and a new noise coefficient). This makes the task a
    continual generate-and-test / feature-search problem.

    Unlike the JAX version (which is a pure ``eqx.Module`` whose ``step``
    returns a new task state), this version is stateful: ``step`` mutates the
    task's RNG and feature state in place and returns only the sample.
    """

    def __init__(
        self,
        n_target_features: int = 10,
        n_learner_features: int = 20,
        target_noise_std: float = 0.0,
        flip_period: int = -1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_target_features: Number of features in the target network.
            n_learner_features: Number of candidate features the learner sees.
            target_noise_std: Standard deviation of the noise added to the target output.
            flip_period: If > 0, the sign of one randomly chosen target weight is
                flipped every ``flip_period`` steps (including the first step).
                Disabled when <= 0.
            seed: Random seed for reproducibility.
        """
        self.n_target_features = n_target_features
        self.n_learner_features = n_learner_features
        self.target_noise_std = target_noise_std
        self.flip_period = flip_period
        self.step_idx = 0

        self.rng = np.random.default_rng(seed)

        # Initialize learner features (a target index + noise coefficient per slot).
        self.learner_feature_idxs, self.noise_coefficients = self._generate_new_features(
            self.n_learner_features,
        )

        # Initialize target weights, random -1 or +1.
        self.weights = self.rng.integers(0, 2, size=self.n_target_features) * 2 - 1

    def _generate_new_features(
        self, n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ``n`` new feature indices and noise coefficients.

        Returns:
            Tuple containing:
            - Array of shape (n,) of target feature indices.
            - Array of shape (n,) of noise coefficients (uniform in [0, 1]).
        """
        feature_idxs = self.rng.integers(0, self.n_target_features, size=n)
        noise_coefficients = self.rng.uniform(0, 1, size=n)
        return feature_idxs, noise_coefficients

    def step(
        self,
        prune_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Generate a new sample from the feature sifting task.

        Args:
            prune_mask: Boolean/int array of shape (n_learner_features,); truthy
                entries mark candidate slots to discard and regenerate.

        Returns:
            Tuple containing:
            - learner_features: Array of shape (n_learner_features,), the noisy
              features observed by the learner.
            - target_output: Scalar float, the weighted sum of the true target
              features (plus optional noise).
        """
        prune_mask = np.asarray(prune_mask, dtype=bool)

        # Periodically flip the sign of one randomly chosen target weight.
        if self.flip_period > 0 and (self.step_idx % self.flip_period) == 0:
            flip_idx = self.rng.integers(0, self.n_target_features)
            self.weights[flip_idx] = -self.weights[flip_idx]
        self.step_idx += 1

        # Generate fresh features only for the pruned slots and swap them in.
        n_pruned = int(prune_mask.sum())
        if n_pruned > 0:
            new_feature_idxs, new_noise_coefficients = self._generate_new_features(n_pruned)
            self.learner_feature_idxs[prune_mask] = new_feature_idxs
            self.noise_coefficients[prune_mask] = new_noise_coefficients

        # Generate target features and output.
        target_features = self.rng.uniform(-1, 1, size=self.n_target_features)
        target_output = float(np.sum(target_features * self.weights))
        if self.target_noise_std > 0:
            target_output += float(self.rng.normal(0, self.target_noise_std))

        # Generate learner features (clean target value mixed with pure noise).
        learner_features = target_features[self.learner_feature_idxs]
        feature_noise = self.rng.uniform(-1, 1, size=self.n_learner_features)
        learner_features = (
            (1 - self.noise_coefficients) * learner_features
            + self.noise_coefficients * feature_noise
        )

        return learner_features, target_output
