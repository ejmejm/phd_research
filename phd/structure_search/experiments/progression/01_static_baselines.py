"""Step 1 — Static, frozen-connectivity baselines.

Three variants at matched ~30k-connection parameter budget:
- ``dense``           — one-hidden-layer MLP, n_tasks=1 view of the 1568-dim input
- ``block_sparse``    — one sub-MLP per task (n_tasks=2)
- ``random_sparsity`` — DynamicNetwork with per-unit n_in ~ U[1,128], n_out ~ U[1,20]

Sweeps LR × variant over the standard 260417 grid. Expected ordering on
final loss: block_sparse > random_sparsity > dense (tighter oracle wins;
random-sparsity falls between because it shares the dense input space
while still getting random connection luck).
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
    LR_GRID, MLFLOW_PROJECT, build_step1_config,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
)
from phd.structure_search.train import run_config


SWEEP_CONFIG = {
    'experiment': MLFLOW_PROJECT,
    'sweep_name': 'step1_static_baselines',
    'algorithm': 'grid',
    'optuna_storage': resolve_optuna_tracking_uri(),
    'mlflow_storage': resolve_mlflow_tracking_uri(),
    'output_dir': 'output/step1',
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
    cfg = build_step1_config(variant=variant, lr=lr)
    return run_config(cfg)


if __name__ == '__main__':
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
