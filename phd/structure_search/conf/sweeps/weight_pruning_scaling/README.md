## Replication

Stationary 4-task experiments with test-set evaluation every 10k steps.
Compared to `weight_pruning_lr_sweep/01_stationary` (2 tasks, ~9528 weights),
these double the task count to 4 and double the parameter budget to ~19056
weights to keep per-task capacity constant. Commands run from
`phd/structure_search/`.

##### 1. Stationary 4-task: block-sparse vs dense scaling

Block-sparse runs at the 2x budget (24 units). Dense sweeps over budgets
{1x, 2x, 4x, 8x, 16x, 32x} of the new base = {6, 12, 24, 48, 96, 192} units.

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/sweep/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/sweep/dense.yaml
```

After identifying the best LR per variant (block-sparse: one LR; dense: one
LR per budget), update the corresponding `best/` config and run the fixed-LR
follow-ups (30 seeds each):

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/dense_1x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/dense_2x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/dense_4x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/dense_8x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/dense_16x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/dense_32x.yaml
```

##### 2. Nonstationary 4-task: block-sparse vs dense scaling

Same setup as (1) but with `task.permute_period: 4000` (inputs are permuted
every 4000 steps).

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/sweep/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/sweep/dense.yaml
```

After identifying the best LR per variant, update the corresponding `best/`
config and run the fixed-LR follow-ups (30 seeds each):

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/dense_1x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/dense_2x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/dense_4x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/dense_8x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/dense_16x.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/dense_32x.yaml
```
