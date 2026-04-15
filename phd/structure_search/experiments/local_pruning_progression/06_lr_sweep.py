"""Step 6 — LR sweep for statistical-threshold pruning on normalized MNIST.

The step 6 formula assumes σ_x = 1, which required per-pixel standardization
of MNIST. That normalization makes input magnitudes ~3× larger than raw
[0,1] MNIST, so gradients are ~3× larger and the step 4 best LR of 2^-5
is now too high (weights blow up → loss ~7). This sweep re-tunes LR for
the normalized-input regime; the main step 6 CI×SPP sweep then uses the
winner.

Fixed ci=0.95 (middle of the target grid), spp=50, signed utility,
statistical threshold, 20 seeds, 225k steps. Grid extends downward from
step 4's best: {2^-9, 2^-8, 2^-7, 2^-6, 2^-5}.
"""

import scipy.stats
from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    load_mnist_normalized, log_result_metrics_step4,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
    run_statistical_variant,
)

load_mnist_normalized()

CI = 0.95
SPP = 50

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step6_lr_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step6_lr",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        "lr": [2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "separation_f1",
                    "final_budget", "converge_step"],
        "sensitivity": {
            "params": ["lr"],
        },
    },
}


def objective(**params):
    lr = float(params["lr"])
    z_alpha = float(scipy.stats.norm.ppf(CI))
    r = run_statistical_variant(lr, z_alpha=z_alpha, spp=SPP)
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
