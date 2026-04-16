"""Step 7 — LR sweep for demand-driven generation on normalized MNIST.

Step 6's best LR (2^-9) was tuned on a fully-connected init. Step 7 starts
sparse at budget=1500 — same as steps 1-3 — so the gradient norm per step
is much smaller than fully-connected, and the optimal LR is likely larger.
This sweep re-tunes LR before the main allocation-method comparison.

Fixed allocation_method=clipped_linear (we sweep both in 07_generation.py;
clipped is fine for LR identification because both methods drive the same
prune+train dynamics — only the generation-target distribution differs).
CI=0.9, SPP=50, 20 seeds, 225k steps. Grid: {2^-9 ... 2^-3}.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    load_mnist_normalized, log_result_metrics_step7,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
    run_generation_variant,
)

load_mnist_normalized()

ALLOCATION = "clipped_linear"
CI = 0.9
SPP = 50

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step7_lr_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step7_lr",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        "lr": [2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "separation_f1"],
        "sensitivity": {
            "params": ["lr"],
        },
    },
}


def objective(**params):
    lr = float(params["lr"])
    r = run_generation_variant(
        lr=lr, allocation_method=ALLOCATION, ci=CI, spp=SPP)
    log_result_metrics_step7(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
