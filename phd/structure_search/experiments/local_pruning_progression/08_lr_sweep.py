"""Step 8 — LR sweep for statistical-threshold pruning with BCE utility.

Same pipeline as step 6 (fully-connected init, per-weight statistical
confidence threshold, bias-corrected EMA, normalized MNIST, 3-consecutive-
zero stopping) but swaps the signed softmax-error utility for a per-target
BCE utility. BCE's gradient scale differs from signed utility's so the
step 6 best LR (2^-9) is not guaranteed to transfer; this sweep re-tunes
on the same grid before the main CI × SPP sweep.

Fixed ci=0.95, spp=50, 20 seeds, 225k steps.
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
    "sweep_name": "step8_lr_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step8_lr",
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
    r = run_statistical_variant(lr, z_alpha=z_alpha, spp=SPP,
                                utility_fn='bce')
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
