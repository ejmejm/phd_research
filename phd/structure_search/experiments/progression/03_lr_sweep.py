"""Step 3 — Mixed Feature Search LR sweep.

Half of generated units use within-column connectivity (column_generate),
half use random-sparsity (free_generate_protected with
random_input_count=True). 5 seeds per trial; 03_main.py re-runs the
winner at 20 seeds.
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
    BASE_SEED, LR_GRID, MLFLOW_PROJECT, build_step_2_4_config,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
)
from phd.structure_search.experiments.column_guided_search import run_config


SWEEP_SEEDS = list(range(BASE_SEED, BASE_SEED + 5))


SWEEP_CONFIG = {
    'experiment': MLFLOW_PROJECT,
    'sweep_name': 'step3_lr_sweep',
    'algorithm': 'grid',
    'optuna_storage': resolve_optuna_tracking_uri(),
    'mlflow_storage': resolve_mlflow_tracking_uri(),
    'output_dir': 'output/step3_lr_sweep',
    'spec': {
        'direction': 'minimize',
        'metric': 'asymptotic_loss',
    },
    'parameters': {
        'lr': [float(v) for v in LR_GRID],
    },
    'plots': ['sensitivity', 'best_hyperparameters'],
    'plot_params': {
        'metrics': ['asymptotic_loss', 'asymptotic_accuracy', 'average_loss'],
        'sensitivity': {'params': ['lr']},
    },
}


def objective(**params):
    lr = float(params['lr'])
    cfg = build_step_2_4_config(
        variant='mixed_generation', tracker_mode='unit',
        lr=lr, random_input_count=True, seeds=SWEEP_SEEDS,
    )
    return run_config(cfg)


if __name__ == '__main__':
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
