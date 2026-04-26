"""Step 6 — statistical-confidence threshold pruning.

Step 5 showed that more warmup between prune events gives lower loss and
higher F1 — but the right amount of warmup is a hand-tuned hyperparameter
that depends on LR, utility noise, and the total step budget. Step 6
replaces the fixed τ ≤ 0 rule with a per-weight statistical confidence
threshold

    τ_w(t) = −z_α · |w| · σ_x · sqrt(K · (1 + β^t) / (1 − β^t))

where K = (1 − 2/π)(1 − β)/(1 + β) is precomputed and σ_x = 1 after
upfront per-pixel normalization of MNIST. The EMA utility is bias-corrected
as U_corr = U / (1 − β^t). A weight is only pruned when U_corr < τ_w, i.e.
when we're 1 − α confident the negative utility isn't explained by noise.

Hypothesis: with a well-chosen CI, the dependence on SPP should shrink —
the confidence check itself prevents premature pruning of weights that
haven't had time to build a reliable signal.

Grid: CI ∈ {0.9, 0.95, 0.99} × SPP ∈ {50, 100, 200, 400}. LR fixed at
2^-5 (best from step 4). Total training steps held constant at 225k
across all SPPs. CI is the plot hue.
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

LR = 2**-9  # best from 06_lr_sweep on normalized MNIST (see README)

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step6_statistical_threshold_v2",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step6_v2",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        "ci": [0.9, 0.95, 0.99],
        "spp": [50, 100, 200, 400],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "separation_f1",
                    "kept_same_task", "kept_cross_task",
                    "final_budget", "converge_step"],
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
    r = run_statistical_variant(LR, z_alpha=z_alpha, spp=spp)
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
