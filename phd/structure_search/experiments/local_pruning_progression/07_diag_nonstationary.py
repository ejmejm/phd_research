"""Step 7 diagnostic A — non-stationarity response.

Hypothesis: when task 0's labels permute (every 10k steps), there is a
short spike in pruning and generation for task 0's output units, while
task 1 stays stable. After the spike, both tasks return to similar
steady-state rates.

Setup: always permute task 0 only, every 10k steps. Log every 500 steps
(20 windows between each permutation event). Budget=1500, CI=0.9,
SPP=50, lr=2^-6, clipped_linear allocation, 20 seeds, 225k steps.
"""

import mlflow
from common import (
    load_mnist_normalized, log_result_metrics_step7_diag,
    resolve_mlflow_tracking_uri, run_generation_diag,
)

load_mnist_normalized()

LR = 2**-6
CI = 0.9
SPP = 50
PERMUTE_PERIOD = 10_000

if __name__ == "__main__":
    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri())
    mlflow.set_experiment("local_pruning_progression")
    with mlflow.start_run(run_name="step7_diag_nonstationary"):
        mlflow.log_params(dict(
            lr=LR, ci=CI, spp=SPP, permute_period=PERMUTE_PERIOD,
            permute_task=0, allocation_method="clipped_linear",
            experiment="diag_nonstationary"))
        r = run_generation_diag(
            lr=LR, ci=CI, spp=SPP,
            allocation_method="clipped_linear",
            permute_task=0, permute_period=PERMUTE_PERIOD)
        log_result_metrics_step7_diag(
            r, t0_label="changed", t1_label="stable")
        print(f"final_loss: {r['final_losses'].mean():.4f}")
        print(f"alignment:  {r['alignments'].mean():.4f}")
