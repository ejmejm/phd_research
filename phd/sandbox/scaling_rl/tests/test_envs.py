"""Smoke tests for stream_ac with each supported gymnax environment.

Covers: CartPole-v1, Breakout-MinAtar, Freeway-MinAtar (discrete action spaces).
Pendulum-v1 (continuous) is excluded until the Gaussian policy branch is added;
see the TODO in train_step.

Run with:
    pytest phd/sandbox/scaling_rl/tests/test_envs.py -v
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from phd.sandbox.scaling_rl.core.envs import get_env_specs, make_env
from phd.sandbox.scaling_rl.scripts.stream_ac import prepare_experiment, train_step


DISCRETE_ENVS = [
    'CartPole-v1',
    'Breakout-MinAtar',
    'Freeway-MinAtar',
]

N_STEPS = 10


def _make_cfg(env_name: str):
    return OmegaConf.create({
        'env': {'name': env_name},
        'model': {
            'hidden_dim': 32,
            'n_layers': 2,
            'activation': 'leaky_relu',
            'weight_init_method': 'sparse',
        },
        'stream_ac': {
            'lr': 1.0,
            'gamma': 0.99,
            'lamda': 0.8,
            'kappa_policy': 3.0,
            'kappa_value': 2.0,
            'entropy_coeff': 0.01,
        },
    })


@pytest.mark.parametrize('env_name', DISCRETE_ENVS)
def test_prepare_experiment(env_name):
    """prepare_experiment returns correctly shaped components."""
    cfg = _make_cfg(env_name)
    train_state, env_state, obs, env, env_params = prepare_experiment(cfg, seed=0)

    assert not train_state.is_continuous
    assert obs.shape == env.obs_shape
    assert train_state.obs_flat_dim == int(np.prod(env.obs_shape))


@pytest.mark.parametrize('env_name', DISCRETE_ENVS)
def test_train_step_runs(env_name):
    """train_step executes N_STEPS without error and produces finite outputs."""
    cfg = _make_cfg(env_name)
    train_state, env_state, obs, env, env_params = prepare_experiment(cfg, seed=42)

    step_fn = jax.jit(partial(train_step, env=env, env_params=env_params))

    for i in range(N_STEPS):
        train_state, env_state, obs, stats = step_fn(train_state, env_state, obs)

    assert int(train_state.step) == N_STEPS
    assert jnp.isfinite(stats.reward)
    assert jnp.isfinite(stats.td_error)
    # episode_return is NaN when no episode ended; finite otherwise
    assert stats.episode_return.dtype == jnp.float32
    assert obs.shape == env.obs_shape


@pytest.mark.parametrize('env_name', DISCRETE_ENVS)
def test_obs_preprocessing(env_name):
    """Actor preprocess_obs produces a 1-D vector matching obs_flat_dim."""
    cfg = _make_cfg(env_name)
    train_state, _, obs, env, _ = prepare_experiment(cfg, seed=0)

    proc = train_state.actor.preprocess_obs(obs)
    assert proc.ndim == 1
    assert proc.shape[0] == train_state.obs_flat_dim == int(np.prod(env.obs_shape))
