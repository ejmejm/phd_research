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
from hybrid import Hybrid
from lms import LMS


SCAN_UNROLL = 4


# ---------------------------------------------------------------------------
# Methods (premade algorithms; each method's hyperparameter defaults live in
# its own file under `DEFAULTS`).
# ---------------------------------------------------------------------------

METHODS = {
    'lms': LMS,
    'cbp_autostep': CBPAutostep,
    'hybrid': Hybrid,
}


class TrainState(eqx.Module):
    task: FeatureSiftingTask
    method: eqx.Module
    step: jax.Array = jnp.array(0)


class StepMetrics(eqx.Module):
    """Metrics collected from a single training step."""
    loss: jax.Array
    baseline_loss: jax.Array        # loss of the always-predict-0 model (= y^2)
    best_possible_noise: jax.Array  # sum over targets of the lowest candidate noise coeff
    n_pruned: jax.Array
    n_generated: jax.Array
    n_pruned_best: jax.Array        # pruned features that were the best for their target
    n_generated_best: jax.Array     # generated features that became the best for their target


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

def _per_target_min_noise(idxs: jax.Array, noise: jax.Array, n_target: int) -> jax.Array:
    """Lowest noise coefficient among the learner features mapped to each target
    feature; 1.0 (pure noise) where a target has no candidate. Noise is in [0, 1],
    so the 1.0 fill never beats a real candidate."""
    return jnp.full(n_target, 1.0).at[idxs].min(noise)


def train_step(train_state: TrainState, _) -> Tuple[TrainState, StepMetrics]:
    """Single online step: the task emits a sample, the method learns from it."""
    old_task = train_state.task
    prune_mask = train_state.method.prune_mask
    n_target = old_task.n_target_features

    # Prune quality: was each pruned feature the lowest-noise candidate for its target,
    # in the (pre-step) state the prune decision acted on?
    old_min = _per_target_min_noise(
        old_task.learner_feature_idxs, old_task.noise_coefficients, n_target)
    old_is_best = old_task.noise_coefficients == old_min[old_task.learner_feature_idxs]

    # Step the task (regenerates pruned slots) and the method.
    task, (x, y) = old_task.step(prune_mask)
    method, loss = train_state.method.step(x, y)

    # Generation quality + best-achievable coverage, in the post-step state.
    new_min = _per_target_min_noise(
        task.learner_feature_idxs, task.noise_coefficients, n_target)
    new_is_best = task.noise_coefficients == new_min[task.learner_feature_idxs]

    metrics = StepMetrics(
        loss=loss,
        baseline_loss=y ** 2,
        best_possible_noise=new_min.sum(),
        n_pruned=prune_mask.sum(),
        n_generated=prune_mask.sum(),
        n_pruned_best=(prune_mask & old_is_best).sum(),
        n_generated_best=(prune_mask & new_is_best).sum(),
    )

    new_state = tree_replace(
        train_state,
        task=task,
        method=method,
        step=train_state.step + 1,
    )
    return new_state, metrics


def train_multi_step(train_state: TrainState, n_steps: int) -> Tuple[TrainState, StepMetrics]:
    """Run `n_steps` of training via a single jax.lax.scan.

    Returns the final state and per-step metrics stacked along a leading axis
    of length `n_steps`.
    """
    return jax.lax.scan(train_step, train_state, length=n_steps, unroll=SCAN_UNROLL)


def compute_metrics(m: StepMetrics) -> dict:
    """Aggregate a logged cycle's per-step metrics over steps and seeds.

    Every field has a leading seed axis: (n_seeds, log_freq, ...). Losses are
    averaged; pruned/generated counts are summed; the "best" percentages pool
    every prune/generate event over the window and seeds.
    """
    n_seeds = m.loss.shape[0]
    flat = lambda a: np.asarray(a).reshape(n_seeds, -1)   # (n_seeds, log_freq)

    per_seed_loss = flat(m.loss).mean(axis=1)             # (n_seeds,)
    mean_loss = float(per_seed_loss.mean())
    mean_baseline = float(flat(m.baseline_loss).mean())

    total_pruned = float(np.asarray(m.n_pruned).sum())
    total_generated = float(np.asarray(m.n_generated).sum())
    total_pruned_best = float(np.asarray(m.n_pruned_best).sum())
    total_generated_best = float(np.asarray(m.n_generated_best).sum())
    frac = lambda num, den: num / den if den > 0 else float('nan')

    return {
        'loss': mean_loss,
        'loss_std': float(per_seed_loss.std()),
        'normalized_loss': mean_loss / mean_baseline,
        'best_possible_noise': float(flat(m.best_possible_noise).mean()),
        'n_pruned': total_pruned / n_seeds,            # per-seed total over the cycle
        'n_generated': total_generated / n_seeds,
        'pruned_best_frac': frac(total_pruned_best, total_pruned),
        'generated_best_frac': frac(total_generated_best, total_generated),
    }


def run_experiment(cfg: DictConfig, train_state: TrainState, train_fn: Callable) -> TrainState:
    """Outer Python loop calling `train_fn` once per logged cycle."""
    log_freq = cfg.train.log_freq
    num_scans = cfg.train.total_steps // log_freq

    pbar = tqdm(total=cfg.train.total_steps, desc='Training')

    losses = []
    for _ in range(num_scans):
        train_state, step_metrics = train_fn(train_state, log_freq)

        step = int(train_state.step[0])
        metrics = compute_metrics(step_metrics)
        log_metrics(metrics, cfg, step=step)
        losses.append(metrics['loss'])

        pbar.update(log_freq)
        pbar.set_postfix({'loss': f'{metrics["loss"]:.4f}',
                          'norm': f'{metrics["normalized_loss"]:.3f}'})

    pbar.close()

    # End-of-run summary metrics, logged once (no step). Cycles are equal-length, so the
    # mean of per-cycle losses is the per-step average, and the last 10% of cycles is the
    # loss over the last 10% of steps.
    if losses:
        n_tail = max(1, len(losses) // 10)
        log_metrics({
            'average_loss': float(np.mean(losses)),
            'final_loss': float(np.mean(losses[-n_tail:])),
        }, cfg)

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
