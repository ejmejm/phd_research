"""Wiring a config into a runnable experiment, and summarizing the result.

``experiments/train.py`` is a thin Hydra shell over ``run_config`` here.
"""

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from .algorithms import build_algorithm
from .algorithms.base import ConnectivityAlgorithm
from .data import MultiMNISTStream, load_dataset, standardize
from .logging import (
    bind_to_active_run, finish_child_runs, import_logger, init_child_runs,
    log_child_metrics, log_metrics, log_np_array,
)
from .metrics import DIVERGENCE_LOSS_THRESHOLD
from .optim import prepare_optimizer
from .training import TrainState, run_experiment
from .utils import configure_jax, count_params, rng_from_string, set_seed, stack_pytrees


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _dynamic_network_diagnostics(model, n_tasks: int) -> Dict[str, Any]:
    """Active units and connections for a seed-vmapped sparse network."""
    network = model.network
    active_conns = (network.input_indices >= 0).sum(axis=(-1, -2, -3))
    active_out = network.output_mask.astype(bool).sum(axis=(-1, -2))
    return {
        'active_units': network.unit_mask.sum(axis=(-1, -2)),
        'active_connections': active_conns + active_out,
    }


def _build_model_and_specs(cfg: DictConfig, input_dim: int, output_dim: int,
                           n_tasks: int, key):
    """Build the model named by ``model.type``, with its filter and diagnostics.

    Returns ``(model, filter_spec, diagnostics_fn)``.
    """
    model_type = str(cfg.model.get('type', 'padded_mlp'))

    if model_type == 'padded_mlp':
        from .models.padded_mlp import (
            init_model, model_filter_spec, structure_diagnostics)
        model = init_model(cfg, input_dim, output_dim, n_tasks, key=key)
        if str(cfg.model.init_strategy) == 'sparse':
            # The realized Erdos-Renyi count is random, so report it against
            # the budget the width was derived from rather than assume it.
            realized = int(model.w1_mask.sum() + model.w2_mask.sum())
            budget = int(cfg.model.sparse.connection_budget)
            print(f'sparse init: hidden={int(cfg.model.initial_hidden_units)} '
                  f'epsilon={float(cfg.model.sparse.epsilon):.4f} '
                  f'connections={realized} (budget {budget}, '
                  f'{100.0 * realized / budget:.1f}%)')
        return model, model_filter_spec(model), structure_diagnostics

    if model_type == 'dynamic_network':
        from .metrics import compute_structure_metrics
        from .models.sparse_init import (
            derive_sizes, init_sparse_model, model_filter_spec)
        hidden, max_conns, max_fan_out, p_w1, p_w2 = derive_sizes(
            cfg, input_dim, output_dim)
        print(f'sparse network: hidden={hidden} max_conns_per_unit={max_conns} '
              f'max_fan_out={max_fan_out} p_w1={p_w1:.5f} p_w2={p_w2:.5f}')
        model = init_sparse_model(
            cfg, input_dim, output_dim, hidden, max_conns, max_fan_out,
            p_w1, p_w2, key=key)
        return model, model_filter_spec(model), _dynamic_network_diagnostics

    raise ValueError(
        f'Unknown model.type: {model_type!r}. Known: padded_mlp, dynamic_network.')


def prepare_experiment(cfg: DictConfig, algorithm: ConnectivityAlgorithm):
    """Build the seed-stacked train state, the data streams, and problem sizes."""
    seeds = cfg.seed
    n_tasks = int(cfg.task.n_tasks)
    permute_period = int(cfg.task.permute_period)
    do_standardize = bool(cfg.task.get('standardize', False))
    eval_freq = int(cfg.train.get('eval_freq', 0))

    raw_train_images, labels, num_classes, input_dim_per_task = load_dataset('train')
    images = standardize(raw_train_images) if do_standardize else raw_train_images

    test_images = test_labels = None
    if eval_freq > 0:
        test_images, test_labels, _, _ = load_dataset('test')
        if do_standardize:
            test_images = standardize(test_images, ref=raw_train_images)

    input_dim = n_tasks * input_dim_per_task
    output_dim = n_tasks * num_classes

    streams, train_states = [], []
    diagnostics_fn = None
    for seed in seeds:
        rng = jax.random.key(seed)
        streams.append(MultiMNISTStream(
            images=images, labels=labels, n_tasks=n_tasks,
            batch_size=cfg.train.batch_size, seed=seed,
            permute_period=permute_period,
            test_images=test_images, test_labels=test_labels,
        ))

        model, filter_spec, diagnostics_fn = _build_model_and_specs(
            cfg, input_dim, output_dim, n_tasks, key=rng_from_string(rng, 'model'))
        optimizer = prepare_optimizer(
            model, cfg.optimizer.name, cfg.optimizer, filter_spec=filter_spec)
        train_states.append(TrainState(
            model=model, optimizer=optimizer,
            step=jnp.array(0),
            rng=rng_from_string(rng, 'train'),
            algo=algorithm.init_state(model, key=rng_from_string(rng, 'algorithm')),
        ))

    n_params = count_params(train_states[0].model)
    print(f'model={cfg.model.get("type", "padded_mlp")} '
          f'algorithm={algorithm.name} stored_params={n_params} seeds={list(seeds)}')

    return (stack_pytrees(train_states), streams, num_classes, n_tasks, diagnostics_fn)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _log_period(metrics: Dict[str, float], per_seed: Dict[str, list],
                cfg: DictConfig, step: int) -> None:
    log_metrics(metrics, cfg, step=step)
    log_child_metrics(per_seed, cfg, step=step)


def _summarize(cfg: DictConfig, curves: Dict[str, list], train_state) -> dict:
    """Build the run summary a sweep optimizes against."""
    losses, accs = curves['loss'], curves['accuracy']
    diverged = bool(
        losses and (not np.isfinite(losses[-1]) or losses[-1] > DIVERGENCE_LOSS_THRESHOLD))

    # "Asymptotic" = the final 10% of log periods. In the stationary problem
    # this measures converged performance; in the non-stationary problem the
    # network never converges, so it measures how fast it re-adapts. That
    # difference is the point of Figure 2.
    n_tail = max(1, len(losses) // 10)
    summary = {
        'average_loss': float(np.mean(losses)),
        'asymptotic_loss': float(np.mean(losses[-n_tail:])),
        'asymptotic_accuracy': float(np.mean(accs[-n_tail:])),
        'diverged': float(diverged),
    }

    if curves['test_loss']:
        n_test_tail = max(1, len(curves['test_loss']) // 10)
        summary['asymptotic_test_loss'] = float(np.mean(curves['test_loss'][-n_test_tail:]))
        summary['asymptotic_test_accuracy'] = float(
            np.mean(curves['test_accuracy'][-n_test_tail:]))
    elif int(cfg.train.get('eval_freq', 0)) > 0:
        # Diverged before the first evaluation. Emit the keys anyway so a
        # sweep backend looking them up finds them instead of erroring.
        summary['asymptotic_test_loss'] = float('nan')
        summary['asymptotic_test_accuracy'] = float('nan')

    return summary


def run_config(cfg: DictConfig) -> dict:
    """Run one configuration end to end and return its summary."""
    configure_jax(cfg)
    import_logger(cfg)
    bind_to_active_run(cfg)

    if cfg.seed is None:
        cfg.seed = [int(np.random.randint(0, 1_000_000_000))]
    elif isinstance(cfg.seed, int):
        cfg.seed = [cfg.seed]
    else:
        cfg.seed = list(cfg.seed)

    if cfg.get('log_individual_seeds', False) and not cfg.get('mlflow', False) \
            and not cfg.get('comet_ml', False):
        raise ValueError('log_individual_seeds requires mlflow or comet_ml logging.')

    set_seed(cfg.seed[0])
    init_child_runs(cfg.seed, cfg)

    algorithm = build_algorithm(cfg)
    train_state, streams, num_classes, n_tasks, diagnostics_fn = prepare_experiment(
        cfg, algorithm)

    train_state, curves = run_experiment(
        cfg, train_state, streams, algorithm, num_classes, n_tasks,
        diagnostics_fn=diagnostics_fn, log_fn=_log_period,
    )

    summary = _summarize(cfg, curves, train_state)
    print(f'Average loss: {summary["average_loss"]:.4f} | '
          f'Asymptotic loss: {summary["asymptotic_loss"]:.4f} | '
          f'Asymptotic acc: {summary["asymptotic_accuracy"]:.4f}')
    if 'asymptotic_test_accuracy' in summary:
        print(f'Asymptotic test loss: {summary["asymptotic_test_loss"]:.4f} | '
              f'Asymptotic test acc: {summary["asymptotic_test_accuracy"]:.4f}')
    log_metrics(summary, cfg)

    _log_per_seed(cfg, curves)
    finish_child_runs(cfg)
    return summary


def _log_per_seed(cfg: DictConfig, curves: Dict[str, list]) -> None:
    """Upload the per-seed curves.

    Comet rate-limits metric writes, so 30 seeds x hundreds of log periods is
    not viable as a live metric stream. One end-of-run array asset gives the
    analysis notebooks everything they need for confidence intervals without
    burning the budget.
    """
    if not curves['per_seed_loss']:
        return

    per_seed_losses = np.stack(curves['per_seed_loss'])
    per_seed_accs = np.stack(curves['per_seed_accuracy'])
    n_tail = max(1, len(curves['loss']) // 10)

    if cfg.get('log_individual_seeds', False):
        log_np_array(per_seed_losses, 'per_seed_losses', cfg)
        log_np_array(per_seed_accs, 'per_seed_accs', cfg)

    child_summary = {
        'average_loss': per_seed_losses.mean(axis=0).tolist(),
        'asymptotic_loss': per_seed_losses[-n_tail:].mean(axis=0).tolist(),
        'asymptotic_accuracy': per_seed_accs[-n_tail:].mean(axis=0).tolist(),
    }

    if curves['per_seed_test_loss']:
        per_seed_test_losses = np.stack(curves['per_seed_test_loss'])
        per_seed_test_accs = np.stack(curves['per_seed_test_accuracy'])
        n_test_tail = max(1, len(curves['per_seed_test_loss']) // 10)
        child_summary['asymptotic_test_loss'] = (
            per_seed_test_losses[-n_test_tail:].mean(axis=0).tolist())
        child_summary['asymptotic_test_accuracy'] = (
            per_seed_test_accs[-n_test_tail:].mean(axis=0).tolist())
        if cfg.get('log_individual_seeds', False):
            log_np_array(per_seed_test_losses, 'per_seed_test_losses', cfg)
            log_np_array(per_seed_test_accs, 'per_seed_test_accs', cfg)

    log_child_metrics(child_summary, cfg)
