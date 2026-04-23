"""Step 10 — LR sweep for 2-layer LTU statistical-threshold pruning.

Architecture: input (1568) → 100 LTUs (5 per output, fixed routing,
trainable scalar output weight per unit) → softmax CE over 20 outputs
on the 2-task multi-MNIST. Each hidden unit starts with 256 random
input candidates; statistical-threshold pruning operates only on
input → hidden weights.

5 variants (one is a no-prune reference):
  - no_prune              : train only, no statistical threshold
  - bce_ltu               : per-weight BCE remove utility with binary per-unit LTU targets
  - bce_ltu_informative   : BCE utility gated by the `informative` mask
  - signed_ltu            : signed utility with binary targets and sigmoid prediction
  - signed_ltu_inf        : signed utility in the ±∞-target limit; u = (2·t−1)·w·x

LR grid: `{2^-11 … 2^-3}`. CI=0.95, SPP=50, 5 seeds, 225k steps.
Primary metric: final_accuracy. Also reports fanin_f1 (per-unit F1 on
each hidden unit's initial 256-candidate pool, averaged across units
with any same-task candidates).
"""

import scipy.stats
from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    load_mnist_normalized, log_result_metrics_step10,
    resolve_mlflow_tracking_uri, resolve_optuna_tracking_uri,
    run_2layer_variant,
)

load_mnist_normalized()

CI = 0.95
SPP = 50
N_SEEDS = 5

VARIANTS = ['no_prune', 'bce_ltu', 'bce_ltu_informative',
            'signed_ltu', 'signed_ltu_inf']

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step10_lr_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step10_lr",
    "spec": {
        "direction": "maximize",
        "metric": "final_accuracy",
    },
    "parameters": {
        "variant": VARIANTS,
        "lr": [2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5,
               2**-4, 2**-3, 2**-2, 2**-1, 1.0],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_accuracy", "final_loss", "fanin_f1",
                    "final_budget", "converge_step"],
        "sensitivity": {
            "params": ["lr"],
            "hue": ["variant"],
        },
    },
}


def objective(**params):
    variant = str(params["variant"])
    lr = float(params["lr"])
    z_alpha = float(scipy.stats.norm.ppf(CI))
    r = run_2layer_variant(lr, z_alpha=z_alpha, utility_fn=variant,
                           n_seeds=N_SEEDS, spp=SPP)
    log_result_metrics_step10(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
