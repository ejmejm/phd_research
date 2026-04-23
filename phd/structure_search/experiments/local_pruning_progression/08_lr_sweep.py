"""Step 8 — LR sweep for statistical-threshold pruning with BCE utility
AND BCE training loss (matched pair).

The first attempt at step 8 paired BCE utility with softmax CE training
loss; that setup collapsed to chance at every LR (see step 8 LR sweep
history under `step8_lr_sweep` and the step 9 `bce_softmax` variant).
Step 9 showed that BCE utility *does* drive useful pruning when paired
with BCE training loss. This sweep re-tunes LR for the BCE-matched pair
under step 6's fully-connected statistical-threshold pipeline.

Fixed ci=0.95, spp=50, 20 seeds, 225k steps. Grid extended up to 2^-3
because BCE-loss gradient magnitude can differ substantially from
softmax CE's, and the step 9 sparse-init best LR (0.64) scales down
roughly 20× for a fully-connected net → ≈ 2^-5, so the optimum is
plausibly higher than step 6's 2^-9. Accuracy is logged alongside loss
so this can be compared cleanly against step 6.
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
    "sweep_name": "step8_bce_lr_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step8_bce_lr",
    "spec": {
        "direction": "maximize",
        "metric": "final_accuracy",
    },
    "parameters": {
        "lr": [2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5,
               2**-4, 2**-3],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_accuracy", "final_loss", "alignment",
                    "separation_f1", "final_budget", "converge_step"],
        "sensitivity": {
            "params": ["lr"],
        },
    },
}


def objective(**params):
    lr = float(params["lr"])
    z_alpha = float(scipy.stats.norm.ppf(CI))
    r = run_statistical_variant(lr, z_alpha=z_alpha, spp=SPP,
                                utility_fn='bce', loss_fn_name='bce')
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
