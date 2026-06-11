"""The feature-sifting task (JAX version).

Self-contained JAX/equinox port of the project's ``FeatureSiftingTask`` for
people who want the jittable/vmappable version of the environment. Requires
``jax``, ``equinox``, and ``jaxtyping``. The pure-numpy version of the same task
(no extra deps) lives in ``feature_sifting.py``.

Unlike the numpy version, this is a pure ``eqx.Module``: ``step`` carries the
RNG in the task state and returns a *new* task rather than mutating in place, so
it composes with ``jax.lax.scan`` / ``jax.vmap``.
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Return a copy of ``tree`` with the given attributes replaced."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)


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

    # Dynamic parameters (weights and state)
    learner_feature_idxs: Int[Array, 'n_learner_features']
    noise_coefficients: Float[Array, 'n_learner_features']
    weights: List[Float[Array, 'n_target_features']]
    target_noise_std: float
    rng: random.PRNGKey

    def __init__(
        self,
        n_target_features: int = 10,
        n_learner_features: int = 20,
        target_noise_std: float = 0.0,
        key: random.PRNGKey = None,
    ):
        """
        Args:
            n_target_features: Number of features in the target network
            n_learner_features: Number of features in the learner network
            target_noise_std: Standard deviation of the noise added to the target features
            key: JAX PRNG key for reproducibility
        """
        self.n_target_features = n_target_features
        self.n_learner_features = n_learner_features
        self.target_noise_std = target_noise_std

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
    ) -> Tuple['FeatureSiftingTask', Tuple[Float[Array, 'n_learner_features'], Float[Array, '']]]:
        """
        Generate a new sample from the feature sifting task.

        Returns:
            Tuple containing:
            - An updated FeatureSiftingTask state with an updated RNG.
            - (learner_features, target_output):
                - learner_features: Array of shape (n_learner_features,), the noisy features sampled by the learner.
                - target_output: Scalar float, the weighted sum of the true target features.
        """
        new_rng, feature_key, x_key, target_noise_key, feature_noise_key = random.split(self.rng, 5)

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
