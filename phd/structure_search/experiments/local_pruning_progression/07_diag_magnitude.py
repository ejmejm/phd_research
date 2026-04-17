"""Step 7 diagnostic B — magnitude asymmetry.

Hypothesis: when task 0's logits are multiplied by 2 (making its softmax
more peaked → larger gradients → faster learning), task 0 should
accumulate higher demand early on because its connections resolve as
useful more quickly. Over time both tasks should converge to similar
demand as the network settles.

Setup: stationary labels (no permutation), task_scales=[2.0, 1.0].
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
    with mlflow.start_run(run_name="step7_diag_magnitude"):
        mlflow.log_params(dict(
            lr=LR, ci=CI, spp=SPP,
            task_scales="2.0,1.0",
            allocation_method="clipped_linear",
            experiment="diag_magnitude"))
        r = run_generation_diag(
            lr=LR, ci=CI, spp=SPP,
            allocation_method="clipped_linear",
            task_scales=[2.0, 1.0])
        log_result_metrics_step7_diag(
            r, t0_label="scaled_2x", t1_label="unscaled")
        print(f"final_loss: {r['final_losses'].mean():.4f}")
        print(f"alignment:  {r['alignments'].mean():.4f}")
