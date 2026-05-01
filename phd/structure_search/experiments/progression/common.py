"""Shared utilities for the structure-search progression.

Thin layer on top of ``phd.structure_search.train.run_config`` and
``phd.structure_search.experiments.column_guided_search.run_config``:
- Hydra ``DictConfig`` factories (build_step1_config, build_step_2_4_config).
- MLflow / Optuna URI resolvers (mirrored from local_pruning_progression).
- ``log_result_metrics_progression(summary)`` which logs the summary returned
  by ``run_config`` to the currently-active MLflow run. Progression sweep
  scripts invoke ``run_config`` + this logger from within their
  mlflow_sweeper ``objective`` function.
"""

import os
import sys
from typing import Any, Dict, Optional

# Make the repo root importable so ``from phd... import ...`` works when
# scripts are invoked from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', '..'))

import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf


# =============================================================================
# Shared constants
# =============================================================================

MLFLOW_PROJECT = 'structure_search_progression'

N_TASKS = 2
NUM_CLASSES = 10
INPUT_DIM = 1568
OUTPUT_DIM = 20

TOTAL_STEPS = 225_000
LOG_FREQ = 1_000
PERMUTE_PERIOD = 4_000

N_SEEDS = 20
BASE_SEED = 260420
SEEDS = list(range(BASE_SEED, BASE_SEED + N_SEEDS))

BUDGET = 30_000           # target active-connection budget for all variants

ACTIVATION = 'leaky_relu'
OPTIMIZER = 'sgd'

# Step-1 hidden_dim choices (empirically tuned, all within 1% of 30k):
H_DENSE = 19              # BlockSparseMLP(n_tasks=1) count_params = 30172
H_BLOCK_SPARSE = 19       # BlockSparseMLP(n_tasks=2) count_params = 30172
UNITS_RANDOM_SPARSITY = 405  # avg count_active_connections over 5 seeds = 30181

# Steps 2-4: DynamicNetwork dimensions
MAX_UNITS = 2000           # 2 * UNITS_RANDOM_SPARSITY (headroom for generation)
MAX_CONNECTIONS_PER_UNIT = 256  # half_conns = 128 → up to 128 incoming per unit
MAX_FAN_OUT = 40               # max_out = 20 → up to 20 outgoing per unit

PRUNE_RATE = 0.0001
PRUNE_FREQUENCY = 100

# LR sweep grid (matches 260417 convention, extended to 2^-1)
LR_GRID = [2**-11, 2**-9, 2**-7, 2**-5, 2**-3]


# =============================================================================
# URI helpers (mirrored from local_pruning_progression/common.py)
# =============================================================================

def resolve_mlflow_tracking_uri() -> str:
    uri = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')
    prefix = 'sqlite:///'
    if uri.startswith(prefix) and not os.path.isabs(uri[len(prefix):]):
        uri = f'sqlite:///{os.path.abspath(uri[len(prefix):])}'
    return uri


def resolve_optuna_tracking_uri() -> str:
    uri = os.environ.get('OPTUNA_TRACKING_URI', 'sqlite:///optuna.db')
    prefix = 'sqlite:///'
    if uri.startswith(prefix) and not os.path.isabs(uri[len(prefix):]):
        uri = f'sqlite:///{os.path.abspath(uri[len(prefix):])}'
    return uri


# =============================================================================
# Config factories
# =============================================================================

def _base_cfg() -> Dict[str, Any]:
    """Shared Hydra fields that every step uses. Sweep scripts patch specific
    fields on top of this dict before wrapping in ``OmegaConf.create``.
    """
    return {
        'device': 'cuda',
        'jax_jit_cache_dir': '/tmp/jax_cache',
        'mlflow': True,
        'wandb': False,
        'comet_ml': False,
        'log_individual_seeds': False,
        'project': MLFLOW_PROJECT,
        'seed': list(SEEDS),
        'dataset': {
            'name': 'parallel_mnist',
            'n_tasks': N_TASKS,
            'permute_period': PERMUTE_PERIOD,
            'permute_stop': 0,
        },
        'train': {
            'batch_size': 1,
            'loss': 'cross_entropy',
            'total_steps': TOTAL_STEPS,
            'log_freq': LOG_FREQ,
            'eval_freq': 0,
        },
        'optimizer': {
            'name': OPTIMIZER,
            'learning_rate': None,  # set by caller
            'weight_decay': 0,
        },
    }


def build_step1_config(variant: str, lr: float,
                       seeds: Optional[list] = None,
                       total_steps: Optional[int] = None) -> DictConfig:
    """Build a DictConfig for a step-1 static-baseline run.

    variant ∈ {'dense', 'block_sparse', 'random_sparsity'}.
    Consumed by ``phd.structure_search.train.run_config``.
    """
    cfg = _base_cfg()
    cfg['seed'] = list(seeds) if seeds is not None else list(SEEDS)
    if total_steps is not None:
        cfg['train']['total_steps'] = total_steps
    cfg['optimizer']['learning_rate'] = float(lr)

    if variant == 'dense':
        cfg['model'] = {
            'type': 'mlp',
            'n_layers': 2,            # 2 weight layers = 1 hidden layer
            'hidden_dim': H_DENSE,
            'weight_init_method': 'lecun_uniform',
            'activation': ACTIVATION,
        }
    elif variant == 'block_sparse':
        cfg['model'] = {
            'type': 'block_sparse',
            'n_layers': 2,
            'hidden_dim': H_BLOCK_SPARSE,
            'weight_init_method': 'lecun_uniform',
            'activation': ACTIVATION,
        }
    elif variant == 'random_sparsity':
        cfg['model'] = {
            'type': 'dynamic',
            'n_layers': 1,            # DynamicNetwork convention: 1 = one hidden layer
            'hidden_dim': UNITS_RANDOM_SPARSITY,
            'max_units_per_layer': UNITS_RANDOM_SPARSITY,
            'max_connections_per_unit': 128,
            'max_fan_out': 20,
            'activation': ACTIVATION,
            'connect_all_to_output': False,
            # No-op when hidden_dim > 0 (the wiring loop overwrites the
            # init-strategy arrays). Only matters for the empty-init path
            # (step 5's planned hidden_dim=0 + init_strategy='empty').
            'init_strategy': 'linear',
            'random_sparsity_init': True,
        }
    else:
        raise ValueError(f'Unknown step-1 variant: {variant}')

    # Step 1: no restructuring
    cfg['structure_tracker'] = {'enabled': False}

    return OmegaConf.create(cfg)


def build_step_2_4_config(
    *,
    variant: str,                 # 'no_column' / 'mixed_generation' / 'no_column' (for step 4)
    tracker_mode: str,            # 'unit' (step 2/3) or 'connection' (step 4)
    lr: float,
    random_input_count: bool = True,
    seeds: Optional[list] = None,
    total_steps: Optional[int] = None,
) -> DictConfig:
    """Build a DictConfig for steps 2-4 (column_guided_search-driven).

    Consumed by ``phd.structure_search.experiments.column_guided_search.run_config``.
    """
    cfg = _base_cfg()
    cfg['seed'] = list(seeds) if seeds is not None else list(SEEDS)
    if total_steps is not None:
        cfg['train']['total_steps'] = total_steps
    cfg['optimizer']['learning_rate'] = float(lr)

    cfg['variant'] = variant
    cfg['init_mode'] = 'random'
    cfg['log_task_affinity'] = False

    cfg['model'] = {
        'type': 'dynamic',
        'n_layers': 1,
        'hidden_dim': UNITS_RANDOM_SPARSITY,       # active units at init
        'max_units_per_layer': MAX_UNITS,
        'max_connections_per_unit': MAX_CONNECTIONS_PER_UNIT,
        'max_fan_out': MAX_FAN_OUT,
        'activation': ACTIVATION,
        'connect_all_to_output': False,
        # No-op when hidden_dim > 0 (the wiring loop overwrites the
        # init-strategy arrays). Only matters for the empty-init path
        # (step 5's planned hidden_dim=0 + init_strategy='empty').
        'init_strategy': 'linear',
        'random_input_count': random_input_count,
        # Start from random-sparsity connectivity so the init matches step 1.
        # Cap at 128 inputs / 20 outputs per unit even though max_conns=256,
        # max_fan_out=40 — the larger structural caps are only for generation
        # headroom (free_generate's half_conns = max_conns // 2 = 128).
        'random_sparsity_init': True,
        'random_sparsity_max_in': 128,
        'random_sparsity_max_out': 1,
    }

    cfg['structure_tracker'] = {
        'enabled': True,
        'mode': tracker_mode,
        'connection_utility_fn': 'contribution',   # only used in connection mode
        'connection_budget': BUDGET,
        'decay_rate': 0.999,
        'maturity_threshold': 1,
        'prune_frequency': PRUNE_FREQUENCY,
        'prune_rate': PRUNE_RATE,
        'max_new_units_per_step': 128,
    }

    return cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)


# =============================================================================
# MLflow logging
# =============================================================================

def log_result_metrics_progression(summary: Dict[str, Any]) -> None:
    """Log a run's summary metrics to the active MLflow run.

    Invoked by each step's sweep ``objective`` after ``run_config`` returns.
    Keys are logged as-is (no CI — 20-seed aggregation already happens inside
    run_config's log_metrics calls; this logger just ensures the headline
    numbers are attached to the sweep trial).
    """
    for key, value in summary.items():
        try:
            mlflow.log_metric(key, float(value))
        except (TypeError, ValueError):
            # Non-numeric metadata — skip.
            pass
