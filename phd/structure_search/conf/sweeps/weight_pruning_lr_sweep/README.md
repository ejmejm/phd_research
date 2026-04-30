## Replication

Below are a set of command for replicating results for this set of experiments. Commands should be run from the `phd/structure_search/` directory.

##### 1. Stationary structure learning vs. baseline step-size sweep

```bash
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/sweep/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/sweep/dense.yaml
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/sweep/random_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/sweep/structure_search.yaml
```

After identifying the best step-size for each, run the fixed-LR follow-ups (30 seeds each):

```bash
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/best/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/best/dense.yaml
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/best/random_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_lr_sweep/01/best/structure_search.yaml
```