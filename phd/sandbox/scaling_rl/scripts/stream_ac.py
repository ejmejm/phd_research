"""
StreamAC experiment with JAX environments (CartPole, Pendulum, MinAtar).

Single script supporting multiple seeds via jax.vmap.
Algorithm: StreamAC + ObGD (online, per-step actor-critic with eligibility traces).

Supported environments (all via gymnax):
  - CartPole-v1          (discrete, 1-D obs)
  - Pendulum-v1          (continuous, 1-D obs)
  - Breakout-MinAtar     (discrete, 3-D obs flattened)
  - Freeway-MinAtar      (discrete, 3-D obs flattened)

Reference: https://github.com/mohmdelsayed/streaming-drl
"""
from functools import partial
import logging
from typing import Any, Callable, Dict, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import rng_from_string
from phd.feature_search.jax_core.utils import tree_replace
from phd.research_utils.logging import finish_experiment, init_experiment, log_metrics
from phd.sandbox.scaling_rl.core.envs import get_env_specs, make_env
from phd.sandbox.scaling_rl.core.models import StreamACNet
from phd.sandbox.scaling_rl.core.normalizers import ObsNormalizer, RewardScaler
from phd.sandbox.scaling_rl.core.optimizers import ObGDOptimizer
from phd.sandbox.scaling_rl.core.utils import configure_jax, stack_pytrees


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training state and per-step stats
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    # Static
    cfg: DictConfig = eqx.field(static=True)
    is_continuous: bool = eqx.field(static=True)   # Pendulum-style vs CartPole-style
    obs_flat_dim: int = eqx.field(static=True)      # flattened obs size (for MinAtar)
    action_dim: int = eqx.field(static=True)        # raw action dim (before 2x for continuous)
    action_scale: float = eqx.field(static=True)    # |max action| for continuous; 1.0 otherwise

    # Networks
    actor: StreamACNet
    critic: StreamACNet

    # Optimizers
    actor_optimizer: ObGDOptimizer
    critic_optimizer: ObGDOptimizer

    # Online normalizers
    obs_normalizer: ObsNormalizer
    reward_scaler: RewardScaler

    # Mutable scalars
    step: Int[Array, '']
    episode_return: Float[Array, '']  # Accumulates raw reward for the current episode
    rng: PRNGKeyArray

    def __init__(
        self,
        cfg,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        obs_normalizer,
        reward_scaler,
        rng,
        is_continuous: bool,
        obs_flat_dim: int,
        action_dim: int,
        action_scale: float,
    ):
        self.cfg = cfg
        self.is_continuous = is_continuous
        self.obs_flat_dim = obs_flat_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.obs_normalizer = obs_normalizer
        self.reward_scaler = reward_scaler
        self.step = jnp.int32(0)
        self.episode_return = jnp.float32(0.0)
        self.rng = rng


class StepStats(eqx.Module):
    reward: Float[Array, '']
    done: Array          # bool scalar
    td_error: Float[Array, '']
    episode_return: Float[Array, '']  # Return at end of episode; NaN otherwise


# ---------------------------------------------------------------------------
# Single environment step
# ---------------------------------------------------------------------------

def train_step(
    train_state: TrainState,
    env_state,
    obs: Float[Array, 'obs_dim'],
    env,
    env_params,
) -> Tuple[TrainState, Any, Float[Array, 'obs_dim'], StepStats]:
    """One environment step: normalize obs/reward, update networks, auto-reset."""
    cfg = train_state.cfg
    rng, action_key, step_key, reset_key = jax.random.split(train_state.rng, 4)

    # ---- Observation preprocessing + normalization ----
    # preprocess_obs handles any model-specific reshaping (e.g. MinAtar (H,W,C) → 1-D for MLPs).
    # Future model types (CNNs) override preprocess_obs for richer pipelines.
    proc_obs = train_state.actor.preprocess_obs(obs)
    norm_obs, obs_normalizer = train_state.obs_normalizer.update_and_normalize(proc_obs)

    # ---- Action selection ----
    if train_state.is_continuous:
        raw_out = train_state.actor(norm_obs)
        mu, pre_std = jnp.split(raw_out, 2)
        std = jax.nn.softplus(pre_std)
        action = mu + std * jax.random.normal(action_key, shape=mu.shape)
    else:
        logits = train_state.actor(norm_obs)
        action = jax.random.categorical(action_key, logits)

    # ---- Environment step ----
    next_obs, next_env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)

    # ---- Auto-reset on episode end ----
    reset_obs, reset_env_state = env.reset(reset_key, env_params)
    next_obs = jnp.where(done, reset_obs, next_obs)
    next_env_state = jax.tree.map(
        lambda r, n: jnp.where(done, r, n), reset_env_state, next_env_state,
    )

    # ---- Normalize next_obs and scale reward ----
    proc_next_obs = train_state.actor.preprocess_obs(next_obs)
    norm_next_obs, obs_normalizer = obs_normalizer.update_and_normalize(proc_next_obs)
    scaled_reward, reward_scaler = train_state.reward_scaler.update_and_scale(reward, done)

    # ---- TD error (stop-gradient through both value estimates) ----
    v_s = jax.lax.stop_gradient(train_state.critic(norm_obs).squeeze())
    v_next = jax.lax.stop_gradient(train_state.critic(norm_next_obs).squeeze())
    td_target = scaled_reward + cfg.stream_ac.gamma * v_next * (1.0 - done.astype(jnp.float32))
    delta = td_target - v_s

    # ---- Actor update ----
    entropy_coeff = cfg.stream_ac.entropy_coeff

    if train_state.is_continuous:
        def actor_loss_fn(actor):
            raw_out = actor(norm_obs)
            mu, pre_std = jnp.split(raw_out, 2)
            std = jax.nn.softplus(pre_std)
            log_prob = -0.5 * jnp.sum(
                ((action - mu) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
            entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * std ** 2))
            return -log_prob - entropy_coeff * entropy * jnp.sign(delta)
    else:
        def actor_loss_fn(actor):
            logits_ = actor(norm_obs)
            log_probs = jax.nn.log_softmax(logits_)
            log_prob_a = log_probs[action]
            probs = jax.nn.softmax(logits_)
            entropy = -jnp.sum(probs * log_probs)
            return -log_prob_a - entropy_coeff * entropy * jnp.sign(delta)

    actor_grads = jax.grad(actor_loss_fn)(train_state.actor)
    actor_updates, new_actor_optimizer = train_state.actor_optimizer.with_update(
        actor_grads, delta, done,
    )
    new_actor = eqx.apply_updates(train_state.actor, actor_updates)

    # ---- Critic update ----
    def critic_loss_fn(critic):
        return -critic(norm_obs).squeeze()  # Gradient of -V(s); ObGD scales by δ

    critic_grads = jax.grad(critic_loss_fn)(train_state.critic)
    critic_updates, new_critic_optimizer = train_state.critic_optimizer.with_update(
        critic_grads, delta, done,
    )
    new_critic = eqx.apply_updates(train_state.critic, critic_updates)

    # ---- Episode return tracking (raw reward, for monitoring) ----
    episode_return = train_state.episode_return + reward
    completed_return = jnp.where(done, episode_return, jnp.nan)
    episode_return = jnp.where(done, jnp.float32(0.0), episode_return)

    new_train_state = tree_replace(
        train_state,
        actor=new_actor,
        critic=new_critic,
        actor_optimizer=new_actor_optimizer,
        critic_optimizer=new_critic_optimizer,
        obs_normalizer=obs_normalizer,
        reward_scaler=reward_scaler,
        step=train_state.step + 1,
        episode_return=episode_return,
        rng=rng,
    )
    step_stats = StepStats(
        reward=reward,
        done=done,
        td_error=delta,
        episode_return=completed_return,
    )
    return new_train_state, next_env_state, next_obs, step_stats


# ---------------------------------------------------------------------------
# Multi-step scan loop
# ---------------------------------------------------------------------------

def train_multi_step(
    train_state: TrainState,
    env_state,
    obs: Float[Array, 'obs_dim'],
    n_steps: int,
    train_step_fn: Callable,
) -> Tuple[TrainState, Any, Float[Array, 'obs_dim'], StepStats]:
    """Run n_steps via jax.lax.scan for JIT efficiency."""

    def _step(carry, _):
        ts, es, ob = carry
        ts, es, ob, stats = train_step_fn(ts, es, ob)
        return (ts, es, ob), stats

    (train_state, env_state, obs), step_stats = jax.lax.scan(
        _step,
        init=(train_state, env_state, obs),
        xs=None,
        length=n_steps,
    )
    return train_state, env_state, obs, step_stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    train_state: TrainState,
    step_stats: StepStats,
) -> Dict[str, Any]:
    """Aggregate per-step stats into loggable scalars."""
    metrics = {
        'step': train_state.step,
        'mean_reward': jnp.mean(step_stats.reward),
        'mean_abs_td_error': jnp.mean(jnp.abs(step_stats.td_error)),
        'mean_episode_return': jnp.nanmean(step_stats.episode_return),
        'n_episodes': jnp.sum(step_stats.done.astype(jnp.int32)),
    }
    return metrics


# ---------------------------------------------------------------------------
# Multi-seed experiment loop
# ---------------------------------------------------------------------------

def run_multiseed_experiment(
    cfg: DictConfig,
    train_fn: Callable,
    metrics_fn: Callable,
    train_states: TrainState,
    env_states,
    obss: Float[Array, 'n_seeds obs_dim'],
    show_progress: bool = True,
) -> Tuple[TrainState, list]:
    """Python-level loop: call train_fn (vmapped over seeds), then log metrics."""
    sequence_length = cfg.train.log_freq
    train_cycles = cfg.train.total_steps // sequence_length

    all_metrics = []

    if show_progress:
        pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    else:
        pbar = None

    # The first call triggers JIT compilation; subsequent calls use the cached artifact.
    for _ in range(train_cycles):
        train_states, env_states, obss, step_stats = train_fn(
            train_states, env_states, obss, sequence_length,
        )

        metrics = metrics_fn(train_states, step_stats)
        metrics_np = {k: np.asarray(v) for k, v in metrics.items()}
        metrics = {k: v.mean(axis=0) for k, v in metrics_np.items()}
        metrics.update({f'{k}_std': v.std(axis=0) for k, v in metrics_np.items()})
        all_metrics.append(metrics)
        log_metrics(metrics, cfg, step=int(train_states.step[0]))

        if pbar is not None:
            ep_ret = metrics.get('mean_episode_return', float('nan'))
            pbar.set_postfix(ep_return=f'{ep_ret:.1f}')
            pbar.update(sequence_length)

    if pbar is not None:
        pbar.close()

    return train_states, all_metrics


# ---------------------------------------------------------------------------
# Experiment preparation
# ---------------------------------------------------------------------------

def prepare_experiment(cfg: DictConfig, seed: int):
    """Initialise all components for a single seed."""
    rng = jax.random.key(seed)
    env, env_params = make_env(cfg)

    obs_flat_dim, action_dim, is_continuous, action_scale = get_env_specs(env, env_params)

    actor_output_dim = 2 * action_dim if is_continuous else action_dim
    actor = StreamACNet(
        input_dim=obs_flat_dim,
        output_dim=actor_output_dim,
        n_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.hidden_dim,
        activation=cfg.model.activation,
        weight_init_method=cfg.model.weight_init_method,
        key=rng_from_string(rng, 'actor'),
    )
    critic = StreamACNet(
        input_dim=obs_flat_dim,
        output_dim=1,
        n_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.hidden_dim,
        activation=cfg.model.activation,
        weight_init_method=cfg.model.weight_init_method,
        key=rng_from_string(rng, 'critic'),
    )

    sa = cfg.stream_ac
    actor_optimizer = ObGDOptimizer(actor, lr=sa.lr, gamma=sa.gamma, lamda=sa.lamda, kappa=sa.kappa_policy)
    critic_optimizer = ObGDOptimizer(critic, lr=sa.lr, gamma=sa.gamma, lamda=sa.lamda, kappa=sa.kappa_value)

    obs_normalizer = ObsNormalizer(obs_dim=obs_flat_dim)
    reward_scaler = RewardScaler(gamma=sa.gamma)

    obs, env_state = env.reset(rng_from_string(rng, 'env_reset'), env_params)

    train_state = TrainState(
        cfg=cfg,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        obs_normalizer=obs_normalizer,
        reward_scaler=reward_scaler,
        rng=rng_from_string(rng, 'train'),
        is_continuous=is_continuous,
        obs_flat_dim=obs_flat_dim,
        action_dim=action_dim,
        action_scale=action_scale,
    )
    return train_state, env_state, obs, env, env_params


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='../conf', config_name='stream_ac', version_base='1.1')
def main(cfg: DictConfig) -> None:
    """Run StreamAC experiment over multiple seeds."""
    from omegaconf import ListConfig
    assert isinstance(cfg.seed, (list, ListConfig)), 'seed must be a list of integers'

    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)

    seeds = list(cfg.seed)
    run_vars = []
    env = env_params = None

    for seed in seeds:
        train_state, env_state, obs, env, env_params = prepare_experiment(cfg, seed)
        run_vars.append((train_state, env_state, obs))

    train_states, env_states, obss = [stack_pytrees(list(x)) for x in zip(*run_vars)]

    train_step_fn = jax.jit(partial(train_step, env=env, env_params=env_params))
    train_fn = jax.jit(
        jax.vmap(
            partial(train_multi_step, train_step_fn=train_step_fn),
            in_axes=(0, 0, 0, None),
        ),
        static_argnames=('n_steps', 'train_step_fn'),
    )
    metrics_fn = jax.jit(jax.vmap(compute_metrics, in_axes=(0, 0)))

    run_multiseed_experiment(cfg, train_fn, metrics_fn, train_states, env_states, obss)

    finish_experiment(cfg)


if __name__ == '__main__':
    main()
