# MLP vs DynamicNetwork IDBD Sweep

Compare MLP and DynamicNetwork using the IDBD optimizer (with autostep) instead
of Adam. Sweep over learning rate and meta learning rate, centered on the best
values from the initial IDBD sweep (2^-7 for both).

## Setup

Same architecture matching as the Adam sweep: 3 hidden layers, ~201K active
connections each.

| Architecture   | hidden_dim / units | Connections |
|----------------|-------------------|-------------|
| MLP            | 64                | ~201K       |
| DynamicNetwork | 1,598/layer       | ~201K       |

5 learning rates x 5 meta learning rates x 3 seeds = 75 trials per architecture.

## Run commands

From the `phd/structure_search/` directory:

```bash
# MLP sweep (75 trials)
mlflow-sweep conf/sweeps/mlp_vs_dynamic_idbd/mlp_idbd_sweep.yaml

# DynamicNetwork sweep (75 trials)
mlflow-sweep conf/sweeps/mlp_vs_dynamic_idbd/dynamic_idbd_sweep.yaml

# Run both sequentially
mlflow-sweep conf/sweeps/mlp_vs_dynamic_idbd/mlp_idbd_sweep.yaml \
             conf/sweeps/mlp_vs_dynamic_idbd/dynamic_idbd_sweep.yaml
```
