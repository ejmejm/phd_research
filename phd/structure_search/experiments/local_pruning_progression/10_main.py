"""Step 10 — main run: variant × SPP at 20 seeds.

Uses per-variant BEST_LR from 10_lr_sweep (extended to include 2^-2, 2^-1, 1.0
so all optima fell inside the grid). LR was chosen by final_accuracy at
5 seeds; the main run re-runs each (variant, SPP) at 20 seeds for tight
error bars on the final numbers.

SPP sweep matches step 5/6 pattern: {50, 100, 200, 400}. Total training
steps held constant at 225k, so n_cycles scales with SPP and each config
sees the same amount of training.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

import scipy.stats

from common import (
    load_mnist_normalized, log_result_metrics_step10,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
    run_2layer_variant,
)

load_mnist_normalized()

CI = 0.95

BEST_LR = {
    'no_prune':            0.25,
    'bce_ltu':             0.03125,
    'bce_ltu_informative': 0.0625,
    'signed_ltu':          0.25,
    'signed_ltu_inf':      0.5,
}

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step10_main_v2",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step10_main",
    "spec": {
        "direction": "maximize",
        "metric": "final_accuracy",
    },
    "parameters": {
        "variant": list(BEST_LR.keys()),
        "spp": [50, 100, 200, 400],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_accuracy", "final_loss", "fanin_f1",
                    "final_budget", "converge_step"],
        "sensitivity": {
            "params": ["spp"],
            "hue": ["variant"],
        },
    },
}


def objective(**params):
    variant = str(params["variant"])
    spp = int(params["spp"])
    lr = BEST_LR[variant]
    z_alpha = float(scipy.stats.norm.ppf(CI))
    import mlflow
    mlflow.log_param("lr", lr)
    r = run_2layer_variant(lr, z_alpha=z_alpha, utility_fn=variant, spp=spp)
    log_result_metrics_step10(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
