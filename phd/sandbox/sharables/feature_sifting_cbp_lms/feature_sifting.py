"""The feature-sifting task (numpy version).

Self-contained numpy port of the project's ``FeatureSiftingTaskNP`` so this demo
can be shared and run with only numpy installed.
"""

from typing import Optional, Tuple

import numpy as np


class FeatureSiftingTaskNP:
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
    continual generate-and-test / feature-search problem.

    This version is stateful: ``step`` mutates the task's RNG and feature
    state in place and returns only the sample.
    """

    def __init__(
        self,
        n_target_features: int = 10,
        n_learner_features: int = 20,
        target_noise_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_target_features: Number of features in the target network.
            n_learner_features: Number of candidate features the learner sees.
            target_noise_std: Standard deviation of the noise added to the target output.
            seed: Random seed for reproducibility.
        """
        self.n_target_features = n_target_features
        self.n_learner_features = n_learner_features
        self.target_noise_std = target_noise_std

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
