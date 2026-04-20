"""Step 4 — Connection-Level Feature Search.

Same network and feature generation as step 2, but pruning operates at the
individual-weight level via ``ConnectionConnectivityManager``
(contribution_connection_utility). Full hidden units are removed only
when all outgoing connections are zero ("dead unit" detection inside
``ConnectionConnectivityManager.modify_structure``).

Hypothesis: connection-level pruning may leave most surviving units with
very few incoming connections (collapsing fan-in). The ``layer_0/
avg_incoming_conns`` metric (already computed in ``metrics.py``) tracks
this directly.
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
    LR_GRID, MLFLOW_PROJECT, build_step_2_4_config,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
)
from phd.structure_search.experiments.column_guided_search import run_config


SWEEP_CONFIG = {
    'experiment': MLFLOW_PROJECT,
    'sweep_name': 'step4_connection_level',
    'algorithm': 'grid',
    'optuna_storage': resolve_optuna_tracking_uri(),
    'mlflow_storage': resolve_mlflow_tracking_uri(),
    'output_dir': 'output/step4',
    'spec': {
        'direction': 'minimize',
        'metric': 'asymptotic_loss',
    },
    'parameters': {
        'lr': [float(v) for v in LR_GRID],
    },
    'plots': ['sensitivity', 'best_hyperparameters'],
    'plot_params': {
        'metrics': ['asymptotic_loss', 'asymptotic_accuracy',
                    'average_loss', 'layer_0/avg_incoming_conns'],
        'sensitivity': {'params': ['lr']},
    },
}


def objective(**params):
    lr = float(params['lr'])
    cfg = build_step_2_4_config(
        variant='no_column', tracker_mode='connection',
        lr=lr, random_input_count=True,
    )
    return run_config(cfg)


if __name__ == '__main__':
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
