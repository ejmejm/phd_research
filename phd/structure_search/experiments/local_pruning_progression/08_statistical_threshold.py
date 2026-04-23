"""Step 8 — CI × SPP sweep for BCE utility + BCE training loss.

Matched BCE pair under step 6's statistical-threshold pipeline. The
first step 8 attempt (BCE utility with softmax CE training loss) failed
at chance; step 9's utility comparison isolated the cause as
utility/loss mismatch, so this version pairs BCE utility with BCE
training loss.

Grid: CI ∈ {0.9, 0.95, 0.99} × SPP ∈ {50, 100, 200, 400}. LR fixed at
the winner of 08_lr_sweep (BCE-loss variant). Total training steps held
constant at 225k across all SPPs. CI is the plot hue.

Accuracy is logged alongside loss so this step is directly comparable
to step 6 on a loss-independent metric (step 6's training loss is
softmax CE, not BCE).
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

LR = 2**-6  # winner of 08_lr_sweep (BCE): acc 0.671, budget 10,233

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step8_bce_statistical_threshold",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step8_bce",
    "spec": {
        "direction": "maximize",
        "metric": "final_accuracy",
    },
    "parameters": {
        "ci": [0.9, 0.95, 0.99],
        "spp": [50, 100, 200, 400],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_accuracy", "final_loss", "alignment",
                    "separation_f1", "final_budget", "converge_step"],
        "sensitivity": {
            "params": ["spp"],
            "hue": ["ci"],
        },
    },
}


def objective(**params):
    ci = float(params["ci"])
    spp = int(params["spp"])
    z_alpha = float(scipy.stats.norm.ppf(ci))
    r = run_statistical_variant(LR, z_alpha=z_alpha, spp=spp,
                                utility_fn='bce', loss_fn_name='bce')
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
