"""Step 5 — SPP sweep for threshold pruning from fully connected.

Hypothesis: longer training between prune events gives the EMA utility
time to develop a reliable signal, reducing unlucky false-negative prunes
from connections that hadn't had a fair chance to show usefulness. With
SPP=50, the first prune happens after just 50 training steps — not much
warmup at all.

Same setup as step 4, but with SPP swept. LR fixed at 2^-5 (best from
step 4). Total training steps held constant at 225k across all SPPs —
so larger SPP = fewer prune events, not more training. Pruning never
happens at step 0: each cycle trains for SPP steps BEFORE pruning.

Grid: powers of 2, {50, 100, 200, 400, 800}.
"""

from omegaconf import OmegaConf
from mlflow_sweeper.config import SweepConfig
from mlflow_sweeper.runner import run_sweep

from common import (
    load_mnist, log_result_metrics_step4, resolve_mlflow_tracking_uri,
    resolve_optuna_tracking_uri, run_threshold_variant,
)

load_mnist()

LR = 2**-5  # best from step 4

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step5_spp_sweep",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step5",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        "spp": [50, 100, 200, 400, 800],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "separation_f1",
                    "final_budget", "converge_step"],
        "sensitivity": {
            "params": ["spp"],
        },
    },
}


def objective(**params):
    spp = int(params["spp"])
    r = run_threshold_variant(LR, spp=spp)
    log_result_metrics_step4(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True)
