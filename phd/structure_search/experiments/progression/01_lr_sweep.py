"""Step 1 — LR sweep over the three static baselines.

Sweeps ``variant × lr`` at **5 seeds per trial** — light enough to scan the
full LR grid cheaply but with enough noise averaging to pick a winner per
variant. The winning LRs are consumed by ``01_main.py`` to re-run each
variant at 20 seeds as an individual MLflow run (not a sweep).

Variants (all at matched ~30k-connection budget):
- ``dense``           — one-hidden-layer MLP, n_tasks=1 view of the 1568-dim input
- ``block_sparse``    — one sub-MLP per task (n_tasks=2)
- ``random_sparsity`` — DynamicNetwork with per-unit n_in ~ U[1,128], n_out ~ U[1,20]
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / '..' / '..' / '..' / '..'))

from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep
from omegaconf import OmegaConf

from common import (
    BASE_SEED, LR_GRID, MLFLOW_PROJECT, build_step1_config,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
)
from phd.structure_search.train import run_config


SWEEP_SEEDS = list(range(BASE_SEED, BASE_SEED + 5))   # 5 seeds for LR scan


SWEEP_CONFIG = {
    'experiment': MLFLOW_PROJECT,
    'sweep_name': 'step1_lr_sweep',
    'algorithm': 'grid',
    'optuna_storage': resolve_optuna_tracking_uri(),
    'mlflow_storage': resolve_mlflow_tracking_uri(),
    'output_dir': 'output/step1_lr_sweep',
    'spec': {
        'direction': 'minimize',
        'metric': 'asymptotic_loss',
    },
    'parameters': {
        'variant': ['dense', 'block_sparse', 'random_sparsity'],
        'lr': [float(v) for v in LR_GRID],
    },
    'plots': ['sensitivity', 'best_hyperparameters'],
    'plot_params': {
        'metrics': ['asymptotic_loss', 'asymptotic_accuracy', 'average_loss'],
        'sensitivity': {
            'params': ['lr'],
            'hue': ['variant'],
        },
    },
}


def objective(**params):
    variant = str(params['variant'])
    lr = float(params['lr'])
    cfg = build_step1_config(variant=variant, lr=lr, seeds=SWEEP_SEEDS)
    return run_config(cfg)


if __name__ == '__main__':
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
