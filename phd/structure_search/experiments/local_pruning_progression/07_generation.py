"""Step 7 — demand-driven connection generation.

Step 6's statistical pruning self-calibrates and is nearly SPP-independent,
but it can only remove harmful connections — it can't grow into useful new
ones. Step 7 adds a **generation policy**: each cycle, after pruning, we
generate exactly n_pruned new connections, allocated across output neurons
in proportion to a per-neuron **demand** estimate `d_i`. Demand for neuron
i is an EMA of the bias-corrected utility of recently-resolved connections
to that neuron — connections that quickly turned out useful or harmful, or
that aged into the EMA's effective sample size. If a neuron's recent
connections were useful, it likely has more reducible error and gets more
new connections.

We sweep two allocation policies:
- clipped_linear: p_i = max(d_i, 0) / sum(max(d, 0))
- softmax: p_i = softmax(d_i / T) with T = 0.01

Setup: budget=1500 (refilled every cycle), CI=0.9, SPP=50, signed utility,
normalized MNIST. 20 seeds × 225k steps. LR identified by 07_lr_sweep.
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

LR = 2**-6  # best from 07_lr_sweep on normalized MNIST at budget=1500
CI = 0.9
SPP = 50

SWEEP_CONFIG = {
    "experiment": "local_pruning_progression",
    "sweep_name": "step7_generation",
    "algorithm": "grid",
    "optuna_storage": resolve_optuna_tracking_uri(),
    "mlflow_storage": resolve_mlflow_tracking_uri(),
    "output_dir": "output/step7",
    "spec": {
        "direction": "minimize",
        "metric": "final_loss",
    },
    "parameters": {
        "allocation_method": ["clipped_linear", "softmax"],
    },
    "plots": ["sensitivity", "best_hyperparameters"],
    "plot_params": {
        "metrics": ["final_loss", "alignment", "separation_f1",
                    "purity"],
        "sensitivity": {
            "params": ["allocation_method"],
        },
    },
}


def objective(**params):
    method = str(params["allocation_method"])
    r = run_generation_variant(
        lr=LR, allocation_method=method, ci=CI, spp=SPP)
    log_result_metrics_step7(r)


if __name__ == "__main__":
    config = SweepConfig.from_dict_config(OmegaConf.create(SWEEP_CONFIG))
    run_sweep(config, objective, n_jobs=1, log_params=True,
              allow_param_change=True)
