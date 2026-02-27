"""
StreamAC experiment with JAX CartPole.

Single script supporting multiple seeds via jax.vmap.
Algorithm: StreamAC + ObGD (online, per-step actor-critic with eligibility traces).

Reference implementations (PyTorch):
  phd/streaming_rl/core/obgd.py
  phd/streaming_rl/core/algorithms/stream_ac.py
"""
from functools import partial
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import equinox as eqx
import gymnax
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import rng_from_string, set_seed
from phd.feature_search.jax_core.models import MLP
from phd.feature_search.jax_core.utils import tree_replace
from phd.research_utils.logging import finish_experiment, init_experiment, log_metrics
from phd.sandbox.scaling_rl.core.optimizers import ObGDOptimizer
from phd.sandbox.scaling_rl.core.utils import configure_jax, stack_pytrees


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training state and per-step stats
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    # Static
    cfg: DictConfig = eqx.field(static=True)

    # Networks
    actor: MLP
    critic: MLP

    # Optimizers
    actor_optimizer: ObGDOptimizer
    critic_optimizer: ObGDOptimizer

    # Mutable scalars
    step: Int[Array, '']
    episode_return: Float[Array, '']  # Accumulates reward for the current episode
    rng: PRNGKeyArray

    def __init__(self, cfg, actor, critic, actor_optimizer, critic_optimizer, rng):
        self.cfg = cfg
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
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
    """One environment step: sample action, update networks, auto-reset."""
    cfg = train_state.cfg
    rng, action_key, step_key, reset_key = jax.random.split(train_state.rng, 4)

    # ---- Action selection (categorical policy) ----
    logits, _ = train_state.actor(obs)
    action = jax.random.categorical(action_key, logits)

    # ---- Environment step ----
    next_obs, next_env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)

    # ---- Auto-reset on episode end ----
    reset_obs, reset_env_state = env.reset(reset_key, env_params)
    next_obs = jnp.where(done, reset_obs, next_obs)
    next_env_state = jax.tree.map(
        lambda r, n: jnp.where(done, r, n), reset_env_state, next_env_state,
    )

    # ---- TD error (stop-gradient through both value estimates) ----
    v_s = jax.lax.stop_gradient(train_state.critic(obs)[0].squeeze())
    v_next = jax.lax.stop_gradient(train_state.critic(next_obs)[0].squeeze())
    td_target = reward + cfg.stream_ac.gamma * v_next * (1.0 - done.astype(jnp.float32))
    delta = td_target - v_s

    # ---- Actor update ----
    entropy_coeff = cfg.stream_ac.entropy_coeff

    def actor_loss_fn(actor):
        logits_, _ = actor(obs)
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
        return -critic(obs)[0].squeeze()  # Gradient of -V(s); ObGD scales by δ

    critic_grads = jax.grad(critic_loss_fn)(train_state.critic)
    critic_updates, new_critic_optimizer = train_state.critic_optimizer.with_update(
        critic_grads, delta, done,
    )
    new_critic = eqx.apply_updates(train_state.critic, critic_updates)

    # ---- Episode return tracking ----
    episode_return = train_state.episode_return + reward
    # Emit the completed return, then reset accumulator on done
    completed_return = jnp.where(done, episode_return, jnp.nan)
    episode_return = jnp.where(done, jnp.float32(0.0), episode_return)

    new_train_state = tree_replace(
        train_state,
        actor=new_actor,
        critic=new_critic,
        actor_optimizer=new_actor_optimizer,
        critic_optimizer=new_critic_optimizer,
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
) -> Tuple[TrainState, StepStats]:
    """Python-level loop: call train_fn (vmapped over seeds), then log metrics."""
    sequence_length = cfg.train.log_freq
    train_cycles = cfg.train.total_steps // sequence_length

    all_metrics = []

    # Warmup JIT
    train_fn(train_states, env_states, obss, sequence_length)

    if show_progress:
        pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    else:
        pbar = None

    for _ in range(train_cycles):
        train_states, env_states, obss, step_stats = train_fn(
            train_states, env_states, obss, sequence_length,
        )

        metrics = metrics_fn(train_states, step_stats)
        metrics = {k: np.asarray(v).mean(axis=0) for k, v in metrics.items()}
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
    env, env_params = gymnax.make(cfg.env.name)

    obs_dim = env.obs_shape[0]
    n_actions = env.num_actions

    actor = MLP(
        input_dim=obs_dim,
        output_dim=n_actions,
        n_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.hidden_dim,
        weight_init_method=cfg.model.weight_init_method,
        activation=cfg.model.activation,
        key=rng_from_string(rng, 'actor'),
    )
    critic = MLP(
        input_dim=obs_dim,
        output_dim=1,
        n_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.hidden_dim,
        weight_init_method=cfg.model.weight_init_method,
        activation=cfg.model.activation,
        key=rng_from_string(rng, 'critic'),
    )

    sa = cfg.stream_ac
    actor_optimizer = ObGDOptimizer(actor, lr=sa.lr, gamma=sa.gamma, lamda=sa.lamda, kappa=sa.kappa_policy)
    critic_optimizer = ObGDOptimizer(critic, lr=sa.lr, gamma=sa.gamma, lamda=sa.lamda, kappa=sa.kappa_value)

    obs, env_state = env.reset(rng_from_string(rng, 'env_reset'), env_params)

    train_state = TrainState(
        cfg=cfg,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        rng=rng_from_string(rng, 'train'),
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
