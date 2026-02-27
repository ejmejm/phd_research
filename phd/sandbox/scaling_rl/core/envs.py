"""Environment utilities for scaling_rl experiments."""
from typing import Tuple

import gymnax
from gymnax.environments.spaces import Box as GymBox
import numpy as np
from omegaconf import DictConfig


def make_env(cfg: DictConfig):
    """Create a gymnax environment and its default params from config."""
    return gymnax.make(cfg.env.name)


def get_env_specs(env, env_params) -> Tuple[int, int, bool, float]:
    """Return environment metadata needed to configure actor/critic/normalizer.

    Returns:
        obs_flat_dim  – flattened observation dimension (int)
        action_dim    – num discrete actions OR continuous action size
        is_continuous – True for Box action spaces (e.g. Pendulum), False for Discrete
        action_scale  – for continuous: |max action|; unused (1.0) for discrete
    """
    obs_flat_dim = int(np.prod(env.obs_shape))
    action_space = env.action_space(env_params)
    is_continuous = isinstance(action_space, GymBox)
    if is_continuous:
        action_dim = int(np.prod(action_space.shape))
        action_scale = float(np.abs(action_space.high).max())
    else:
        action_dim = int(env.num_actions)
        action_scale = 1.0  # unused for discrete
    return obs_flat_dim, action_dim, is_continuous, action_scale
