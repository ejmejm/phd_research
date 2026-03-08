# MLP vs DynamicNetwork Learning Rate Sweep

Sanity check to verify both model types work end-to-end in the training
pipeline, and to get an initial comparison of how a randomly connected sparse
network performs against a standard fully connected MLP.

## Setup

Both architectures use 3 hidden layers, Adam optimizer, and are matched to
have approximately the same number of active connections (~201K). The
DynamicNetwork uses 32 random incoming connections per unit drawn from any
prior layer, with all hidden units connected to the output.

| Architecture   | hidden_dim / units | Connections |
|----------------|-------------------|-------------|
| MLP            | 64                | ~201K       |
| DynamicNetwork | 1,598/layer       | ~201K       |

Learning rates sweep 9 values on a log2 grid from 2^-23 to 2^-7, each with
3 seeds.

## Run commands

From the `phd/structure_search/` directory:

```bash
# MLP sweep (9 lr x 3 seeds = 27 trials)
mlflow-sweep conf/sweeps/mlp_vs_dynamic/mlp_lr_sweep.yaml

# DynamicNetwork sweep (9 lr x 3 seeds = 27 trials)
mlflow-sweep conf/sweeps/mlp_vs_dynamic/dynamic_lr_sweep.yaml

# Run both sequentially
mlflow-sweep conf/sweeps/mlp_vs_dynamic/mlp_lr_sweep.yaml \
             conf/sweeps/mlp_vs_dynamic/dynamic_lr_sweep.yaml
```
