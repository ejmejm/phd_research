"""Online running-statistics normalizers for StreamAC.

Implements observation normalization and reward scaling matching the
reference streaming-drl wrappers (NormalizeObservation, ScaleReward).
All state is carried as JAX arrays inside eqx.Modules so they compose
naturally with lax.scan and vmap.
"""
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from phd.jax_core.utils import tree_replace


class RunningMeanVar(eqx.Module):
    """Welford online mean/variance estimator.

    Works element-wise for arrays of any shape.  Variance is the
    biased (population) estimate: M2 / count.
    """
    count: Array   # int32 scalar
    mean: Array    # float32, same shape as tracked quantity
    M2: Array      # float32, same shape as mean

    def update(self, x: Array) -> 'RunningMeanVar':
        count = self.count + jnp.int32(1)
        delta = x - self.mean
        mean = self.mean + delta / count.astype(jnp.float32)
        delta2 = x - mean
        M2 = self.M2 + delta * delta2
        return tree_replace(self, count=count, mean=mean, M2=M2)

    def var(self) -> Array:
        """Population variance; returns 1.0 before at least 2 samples."""
        return jnp.where(self.count >= 2, self.M2 / self.count.astype(jnp.float32), jnp.ones_like(self.M2))


class ObsNormalizer(eqx.Module):
    """Normalize observations using a running mean and variance.

    Mirrors gymnasium's NormalizeObservation wrapper:
      obs_normalized = (obs - mean) / sqrt(var + epsilon)

    Running stats are updated with every raw observation passed in.
    """
    stats: RunningMeanVar
    epsilon: float = eqx.field(static=True)

    def __init__(self, obs_dim: int, epsilon: float = 1e-8):
        self.stats = RunningMeanVar(
            count=jnp.int32(0),
            mean=jnp.zeros(obs_dim, dtype=jnp.float32),
            M2=jnp.zeros(obs_dim, dtype=jnp.float32),
        )
        self.epsilon = epsilon

    def update_and_normalize(
        self, obs: Float[Array, 'obs_dim']
    ) -> Tuple[Float[Array, 'obs_dim'], 'ObsNormalizer']:
        new_stats = self.stats.update(obs)
        normalized = (obs - new_stats.mean) / jnp.sqrt(new_stats.var() + self.epsilon)
        return normalized, tree_replace(self, stats=new_stats)


class RewardScaler(eqx.Module):
    """Scale rewards by the running standard deviation of the discounted return.

    Mirrors gymnasium's ScaleReward wrapper:
      G_t = gamma * G_{t-1} * (1 - done) + r_t
      r_scaled = r_t / sqrt(var(G) + epsilon)

    Only the variance (not mean) is used for scaling — the mean is not subtracted.
    """
    stats: RunningMeanVar
    return_trace: Array  # float32 scalar — current discounted return estimate
    gamma: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, gamma: float, epsilon: float = 1e-8):
        self.stats = RunningMeanVar(
            count=jnp.int32(0),
            mean=jnp.float32(0.0),
            M2=jnp.float32(0.0),
        )
        self.return_trace = jnp.float32(0.0)
        self.gamma = gamma
        self.epsilon = epsilon

    def update_and_scale(
        self,
        reward: Float[Array, ''],
        done: Array,
    ) -> Tuple[Float[Array, ''], 'RewardScaler']:
        """Update return trace and running stats, return scaled reward."""
        new_trace = (
            self.return_trace * self.gamma * (1.0 - done.astype(jnp.float32)) + reward
        )
        new_stats = self.stats.update(new_trace)
        scaled = reward / jnp.sqrt(new_stats.var() + self.epsilon)
        return scaled, tree_replace(self, stats=new_stats, return_trace=new_trace)
