"""Step 3 — signed utility vs contribution utility across budgets.

Grid: utility_fn × budget. Dynamic variant only, LR fixed at 0.16.
Sensitivity plot shows budget-vs-alignment colored by utility_fn.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    N_CYCLES, N_SEEDS, SPP,
    load_mnist, log_result_metrics, resolve_mlflow_tracking_uri,
    resolve_optuna_tracking_uri, run_variant,
)

load_mnist()

LR = 0.16

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step3_signed_utility",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step3",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        "utility_fn": ["contribution", "signed"],
        "budget": [1500, 500, 150, 50, 20],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "purity"],
        "sensitivity": {
            "params": ["budget"],
            "hue": ["utility_fn"],
        },
    },
}


def objective(**params):
    utility_fn = str(params["utility_fn"])
    budget = int(params["budget"])
    r = run_variant("dynamic", LR, budget=budget, utility_fn=utility_fn)
    log_result_metrics(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
