"""Step 4 — threshold pruning from fully connected, LR sweep.

Start fully connected (31,360 connections). Every 50 steps, prune all
connections with signed EMA utility <= 0. No generation. Continue until
3 consecutive prune events prune 0 connections; then keep training for
the eval window.

Sensitivity plot shows LR vs loss, alignment, final_budget, converge_cycle.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    N_CYCLES, N_SEEDS, SPP,
    load_mnist, log_result_metrics_step4, resolve_mlflow_tracking_uri,
    resolve_optuna_tracking_uri, run_threshold_variant,
)

load_mnist()

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step4_threshold_lr_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step4",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        # Powers of 2 centered on 2^-5 = 0.03125
        "lr": [2**-7, 2**-6, 2**-5, 2**-4, 2**-3],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "final_budget",
                    "converge_cycle"],
        "sensitivity": {
            "params": ["lr"],
        },
    },
}


def objective(**params):
    lr = float(params["lr"])
    r = run_threshold_variant(lr)
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
