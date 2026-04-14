"""Step 1 — LR sweep for linear model with global pruning.

Grid: lr × variant (dynamic, fixed_random, fixed_intask).
Sensitivity plot shows lr-vs-loss colored by variant.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    BUDGET, N_CYCLES, N_SEEDS, SPP,
    load_mnist, log_result_metrics, resolve_mlflow_tracking_uri,
    resolve_optuna_tracking_uri, run_variant,
)

# Pre-load MNIST so every trial reuses the same cached arrays.
load_mnist()

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step1_lr_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step1",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        "variant": ["dynamic", "fixed_random", "fixed_intask"],
        "lr": [1.5625e-4, 6.25e-4, 2.5e-3, 1e-2, 4e-2, 1.6e-1, 6.4e-1],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "purity"],
        "sensitivity": {
            "params": ["lr"],
            "hue": ["variant"],
        },
    },
}


def objective(**params):
    variant = str(params["variant"])
    lr = float(params["lr"])
    r = run_variant(variant, lr, budget=BUDGET)
    log_result_metrics(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
