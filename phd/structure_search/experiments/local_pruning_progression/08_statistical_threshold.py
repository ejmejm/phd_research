"""Step 8 — statistical-confidence threshold pruning with BCE utility.

Mirrors step 6's pipeline and sweep exactly, substituting a per-target
BCE utility for the signed softmax-error utility:

    pre_act_removed = pre_act − h · W
    lp = target·log σ(pre_act) + (1 − target)·log σ(−pre_act)
    lp_removed = target·log σ(pre_act_removed) + (1 − target)·log σ(−pre_act_removed)
    U = (−lp_removed) − (−lp)

The sign convention matches signed utility (positive → removing hurts
loss → keep), so the step 6 threshold formula

    τ_w = −z_α · |w| · σ_x · sqrt(K · (1 + β^t) / (1 − β^t))

transfers unchanged. Training loss remains softmax CE — only the pruning
signal changes. This isolates the utility choice from every other design
decision.

Hypothesis: BCE gives each output head an independent pressure signal.
For a well-trained task-1 output neuron, task-0 inputs receive a stable
0-target BCE gradient that averages toward negative utility more cleanly
than the softmax-error signal (which couples across classes within a
task). Expect higher alignment / F1 than step 6 at comparable loss.

Grid: CI ∈ {0.9, 0.95, 0.99} × SPP ∈ {50, 100, 200, 400}. LR fixed at
the winner from 08_lr_sweep. Total training steps held constant at 225k
across all SPPs.
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

LR = 2**-9  # placeholder — update once 08_lr_sweep finishes

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step8_statistical_threshold",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step8",
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
    r = run_statistical_variant(LR, z_alpha=z_alpha, spp=spp,
                                utility_fn='bce')
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
