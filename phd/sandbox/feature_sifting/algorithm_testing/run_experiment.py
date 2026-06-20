from typing import Callable, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from phd.feature_search.jax_core.experiment_helpers import set_seed
from phd.jax_core.tasks.feature_sifting import FeatureSiftingTask
from phd.jax_core.utils import configure_jax, stack_pytrees, tree_replace
from phd.research_utils.logging import init_experiment, log_metrics, finish_experiment

from cbp_autostep import CBPAutostep
from lms import LMS


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Methods (premade algorithms; each method's hyperparameter defaults live in
# its own file under `DEFAULTS`).
# ---------------------------------------------------------------------------

METHODS = {
    'lms': LMS,
    'cbp_autostep': CBPAutostep,
}


class TrainState(eqx.Module):
    task: FeatureSiftingTask
    method: eqx.Module
    step: jax.Array = jnp.array(0)


class StepMetrics(eqx.Module):
    """Metrics collected from a single training step."""
    loss: jax.Array


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def prepare_experiment(cfg: DictConfig) -> TrainState:
    """Initialize the task and method for each seed, stacked for vmap.

    Returns a single TrainState whose leaves carry a leading seed axis, so the
    whole experiment runs for every seed at once under jax.vmap.
    """
    seeds = cfg.seed
    if seeds is None:
        seeds = [np.random.randint(0, 2**31)]
    elif isinstance(seeds, int):
        seeds = [seeds]
    else:
        seeds = list(seeds)
    set_seed(seeds[0])

    method_cls = METHODS[cfg.method.name]
    hparams = dict(method_cls.DEFAULTS)
    if cfg.method.get('hparams') is not None:
        hparams.update(OmegaConf.to_container(cfg.method.hparams, resolve=True))

    train_states = []
    for seed in seeds:
        task_key, method_key = random.split(random.PRNGKey(seed))
        task = FeatureSiftingTask(**cfg.task, key=task_key)
        method = method_cls.init(task.n_learner_features, hparams, method_key)
        train_states.append(TrainState(task=task, method=method))

    return stack_pytrees(train_states)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(train_state: TrainState, _) -> Tuple[TrainState, StepMetrics]:
    """Single online step: the task emits a sample, the method learns from it."""
    task, (x, y) = train_state.task.step(train_state.method.prune_mask)
    method, loss = train_state.method.step(x, y)

    new_state = tree_replace(
        train_state,
        task=task,
        method=method,
        step=train_state.step + 1,
    )
    return new_state, StepMetrics(loss=loss)


def train_multi_step(train_state: TrainState, n_steps: int) -> Tuple[TrainState, StepMetrics]:
    """Run `n_steps` of training via a single jax.lax.scan.

    Returns the final state and per-step metrics stacked along a leading axis
    of length `n_steps`.
    """
    return jax.lax.scan(train_step, train_state, length=n_steps, unroll=SCAN_UNROLL)


def compute_metrics(step_metrics: StepMetrics) -> dict:
    """Aggregate a logged cycle's per-step metrics across seeds.

    `step_metrics.loss` has a leading seed axis: (n_seeds, log_freq, ...). Take
    each seed's mean over the cycle, then report the mean / std across seeds
    (matching the run-script convention).
    """
    per_seed_loss = step_metrics.loss.reshape(step_metrics.loss.shape[0], -1).mean(axis=1)
    return {
        'loss': float(per_seed_loss.mean()),
        'loss_std': float(per_seed_loss.std()),
    }


def run_experiment(cfg: DictConfig, train_state: TrainState, train_fn: Callable) -> TrainState:
    """Outer Python loop calling `train_fn` once per logged cycle."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq

    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    for _ in range(num_scans):
        train_state, step_metrics = train_fn(train_state, log_freq)

        step = int(train_state.step[0])
        metrics = compute_metrics(step_metrics)
        log_metrics(metrics, cfg, step=step)

        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{metrics["loss"]:.4f}'})

    pbar.close()
    return train_state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='conf', config_name='feature_sifting', version_base='1.1')
def main(cfg: DictConfig) -> None:
    configure_jax(cfg)
    cfg = init_experiment(cfg.project, cfg)

    try:
        train_state = prepare_experiment(cfg)
        train_fn = jax.jit(jax.vmap(train_multi_step, in_axes=(0, None)),
                           static_argnames=('n_steps',))
        train_state = run_experiment(cfg, train_state, train_fn)
    finally:
        finish_experiment(cfg)

    print('Run complete!')


if __name__ == '__main__':
    main()
