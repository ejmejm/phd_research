"""Step 7 diagnostic C — random-label task.

Hypothesis: when task 1's labels are replaced with uniform-random classes
each step, task 1's connections provide no useful gradient signal, so
their utility should be near-zero → lower demand → fewer generated
connections for task 1 → task 0 accumulates more of the budget over time.

Setup: stationary task 0 (no permutation), task 1 gets random labels.
Budget=1500, CI=0.9, SPP=50, lr=2^-6, clipped_linear, 20 seeds, 225k.
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

if __name__ == "__main__":
    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri())
    mlflow.set_experiment("local_pruning_progression")
    with mlflow.start_run(run_name="step7_diag_random_task"):
        mlflow.log_params(dict(
            lr=LR, ci=CI, spp=SPP,
            random_task=1,
            allocation_method="clipped_linear",
            experiment="diag_random_task"))
        r = run_generation_diag(
            lr=LR, ci=CI, spp=SPP,
            allocation_method="clipped_linear",
            random_task=1)
        log_result_metrics_step7_diag(
            r, t0_label="learnable", t1_label="random")
        print(f"final_loss: {r['final_losses'].mean():.4f}")
        print(f"alignment:  {r['alignments'].mean():.4f}")
