"""Shared utilities for the local_pruning_progression experiments.

Intentionally minimal — covers fixed-mask sampling, purity/entropy on
linear masks, MLflow run setup, and a ci95 helper. Per-experiment JAX
code lives in the step scripts.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


MLFLOW_PROJECT = 'local_pruning_progression'
MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db'


# ─── Stats ────────────────────────────────────────────────────────────

def ci95(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    return float(1.96 * arr.std(ddof=0) / np.sqrt(len(arr)))


# ─── Fixed-mask samplers (used for baselines) ─────────────────────────

def sample_fixed_mask_random(key, output_dim: int, input_dim: int,
                             budget: int) -> jnp.ndarray:
    """Uniformly sample `budget` of the output_dim*input_dim possible
    connections. Returns an int32 mask of shape (output_dim, input_dim)."""
    total = output_dim * input_dim
    assert budget <= total, f'budget {budget} exceeds total slots {total}'
    noise = jax.random.uniform(key, (total,))
    # Take the `budget` smallest-noise indices.
    picks = jnp.argsort(noise)[:budget]
    flat = jnp.zeros(total, dtype=jnp.int32).at[picks].set(1)
    return flat.reshape(output_dim, input_dim)


def sample_fixed_mask_intask(key, output_dim: int, input_dim: int,
                             budget: int, n_tasks: int = 2) -> jnp.ndarray:
    """Block-diagonal random mask: budget/n_tasks connections sampled
    uniformly within each task's (output_dim/n_tasks × input_dim/n_tasks)
    block. Budget must divide evenly across tasks."""
    assert budget % n_tasks == 0
    assert output_dim % n_tasks == 0 and input_dim % n_tasks == 0
    out_per = output_dim // n_tasks
    in_per = input_dim // n_tasks
    budget_per = budget // n_tasks
    block = out_per * in_per
    assert budget_per <= block

    M = jnp.zeros((output_dim, input_dim), dtype=jnp.int32)
    keys = jax.random.split(key, n_tasks)
    for t in range(n_tasks):
        noise = jax.random.uniform(keys[t], (block,))
        picks = jnp.argsort(noise)[:budget_per]
        flat = jnp.zeros(block, dtype=jnp.int32).at[picks].set(1)
        sub = flat.reshape(out_per, in_per)
        M = M.at[t * out_per:(t + 1) * out_per,
                 t * in_per:(t + 1) * in_per].set(sub)
    return M


def sample_init_mask_dynamic(key, output_dim: int, input_dim: int,
                             budget: int) -> jnp.ndarray:
    """Initial mask for the dynamic method — same as random fixed,
    just gets pruned/regenerated during the run."""
    return sample_fixed_mask_random(key, output_dim, input_dim, budget)


# ─── Purity / entropy on a linear mask ────────────────────────────────

def purity_entropy_linear(M: np.ndarray, input_per_task: int = 784,
                          n_tasks: int = 2) -> tuple:
    """Compute per-seed (purity, entropy).

    M shape: (output_dim, input_dim). For each output neuron, split its
    active inputs by task block and compute purity = max(t_k) / sum(t_k)
    and normalized binary entropy. Units with no active inputs are
    skipped. Averages over units.
    """
    M = np.asarray(M)
    output_dim = M.shape[0]
    purs, ents = [], []
    for u in range(output_dim):
        counts = np.array([
            int(M[u, t * input_per_task:(t + 1) * input_per_task].sum())
            for t in range(n_tasks)
        ])
        total = counts.sum()
        if total == 0:
            continue
        purs.append(counts.max() / total)
        ps = counts / total
        e = 0.0
        for p in ps:
            if p > 0:
                e -= p * np.log2(p)
        ents.append(e / np.log2(n_tasks))
    if not purs:
        return 0.0, 1.0
    return float(np.mean(purs)), float(np.mean(ents))


def batch_purity_entropy_linear(all_M: np.ndarray, input_per_task: int = 784,
                                n_tasks: int = 2):
    """Apply purity_entropy_linear to each seed in a (S, output, input) mask.
    Returns (purs, ents) arrays of shape (S,)."""
    all_M = np.asarray(all_M)
    S = all_M.shape[0]
    purs = np.zeros(S)
    ents = np.zeros(S)
    for s in range(S):
        p, e = purity_entropy_linear(all_M[s], input_per_task, n_tasks)
        purs[s] = p
        ents[s] = e
    return purs, ents


# ─── MLflow ───────────────────────────────────────────────────────────

def mlflow_start(run_name: str, params: Dict[str, Any]):
    """Start an MLflow run under the progression's shared project.

    Returns the mlflow module (already imported) so the caller can
    log further metrics/params. Uses a flat sqlite tracking URI.
    """
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_PROJECT)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({k: str(v) for k, v in params.items()})
    return mlflow


def mlflow_log_aggregate(mlflow, name: str, values: np.ndarray):
    """Log mean and 95% CI of a (S,) seed array as two metrics."""
    values = np.asarray(values)
    mlflow.log_metric(name, float(values.mean()))
    mlflow.log_metric(f'{name}_ci95', ci95(values))


def mlflow_log_trajectory(mlflow, name: str, values: np.ndarray,
                          steps: np.ndarray):
    """Log a mean-across-seeds trajectory, one point per (value, step) pair.

    values: (S, T), steps: (T,)
    """
    values = np.asarray(values)
    steps = np.asarray(steps)
    mean = values.mean(axis=0)
    for t, s in enumerate(steps):
        mlflow.log_metric(name, float(mean[t]), step=int(s))
