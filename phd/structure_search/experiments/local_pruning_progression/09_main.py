"""Step 9 — main run across 4 utility/loss variants × 4 budgets at 20 seeds.

Uses per-(variant, budget) LRs from the stage-1 LR sweep (09_lr_sweep.py)
selected by highest final_accuracy.

Variants (utility / training loss):
  - signed       / softmax_ce   (step 3 baseline, reproduced for comparison)
  - softmax_ce   / softmax_ce   (closed-form softmax-CE LOO utility)
  - bce_softmax  = bce / softmax_ce
  - bce_bce      = bce / bce    (matched BCE pair)

Budget sweep matches step 3: {50, 150, 500, 1500}. 225k steps, SPP=50.
Accuracy is the primary comparison metric since `bce_bce`'s training
loss is on a different scale from the softmax-CE variants.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    log_result_metrics, load_mnist,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri, run_variant,
)

load_mnist()

VARIANT_SPECS = {
    'signed':       ('signed',     'softmax_ce'),
    'softmax_ce':   ('softmax_ce', 'softmax_ce'),
    'bce_softmax':  ('bce',        'softmax_ce'),
    'bce_bce':      ('bce',        'bce'),
}

# Best LR per (method, budget) from step9_lr_sweep_full (5 seeds), chosen
# by final_accuracy. See README Step 9 for sweep context.
BEST_LR = {
    ('signed',       50):   0.64,
    ('signed',      150):   0.64,
    ('signed',      500):   0.16,
    ('signed',     1500):   0.16,
    ('softmax_ce',   50):   0.64,
    ('softmax_ce',  150):   0.64,
    ('softmax_ce',  500):   0.64,
    ('softmax_ce', 1500):   0.64,
    ('bce_softmax',  50):   0.16,
    ('bce_softmax', 150):   0.04,
    ('bce_softmax', 500):   0.04,
    ('bce_softmax',1500):   0.04,
    ('bce_bce',      50):   0.64,
    ('bce_bce',     150):   0.64,
    ('bce_bce',     500):   0.64,
    ('bce_bce',    1500):   0.64,
}

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step9_main",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step9_main",
    "spec": {
        "direction": "maximize",
        "metric": "final_accuracy",
    },
    "parameters": {
        "method": list(VARIANT_SPECS.keys()),
        "budget": [50, 150, 500, 1500],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_accuracy", "final_loss", "alignment",
                    "separation_f1"],
        "sensitivity": {
            "params": ["budget"],
            "hue": ["method"],
        },
    },
}


def objective(**params):
    method = str(params["method"])
    budget = int(params["budget"])
    lr = BEST_LR[(method, budget)]
    utility_fn, loss_fn_name = VARIANT_SPECS[method]
    import mlflow
    mlflow.log_param("lr", lr)
    r = run_variant('dynamic', lr, budget=budget,
                    utility_fn=utility_fn, loss_fn_name=loss_fn_name)
    log_result_metrics(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
