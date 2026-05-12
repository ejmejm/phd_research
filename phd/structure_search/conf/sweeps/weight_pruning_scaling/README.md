## Replication

Three experiments at 4× the per-task parameter budget of
`weight_pruning_lr_sweep` (so block-sparse and dense both target ~19056
total weights at n_tasks=4). All commands run from `phd/structure_search/`.

The general workflow for each numbered experiment is:

1. Run the LR sweeps under `sweep/`.
2. Pick the best LR for each method/budget from the sweep results.
3. Fill in the `???` placeholder under `best/<method>.yaml`'s
   `optimizer.learning_rate` field with that LR.
4. Run the fixed-LR follow-up under `best/` (30 seeds, 101–130).

### 1. Stationary, 4 tasks: block-sparse vs dense scaling

Stationary with test-set evaluation every 10k steps. Block-sparse runs at
the 2× budget (24 units). Dense sweeps over budgets {1×, 2×, 4×, 8×, 16×,
32×, 64×} of the base = {6, 12, 24, 48, 96, 192, 384} units.

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/sweep/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/sweep/dense.yaml
```

Update `optimizer.learning_rate` in each `best/` config, then:

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/01_stationary/best/dense.yaml
```

### 2. Nonstationary, 4 tasks: all four methods

Same setup as (1) but with `task.permute_period: 800` (inputs permuted
every 800 steps) and `train.eval_freq: 0` (evaluation disabled, all data
used for training). Block-sparse, dense, random-sparse, and
structure-search are all included.

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/sweep/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/sweep/dense.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/sweep/random_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/sweep/structure_search.yaml
```

`random_sparse` is itself a sparsity × LR sweep over `initial_hidden_units`
∈ {48, 96, 192, 384}; dense sweeps over the same budgets as in (1).
Update `optimizer.learning_rate` in each `best/` config, then:

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/dense.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/random_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/02_nonstationary/best/structure_search.yaml
```

### 3. Nonstationary, scaling task count: all four methods

Same nonstationary setup as (2), but sweeps `task.n_tasks ∈ {2, 4, 8, 16,
32}` with `permute_period = 3200 // n_tasks` (so 800 at n=4 matches the
(2) setup). Per-task capacity is held constant via formulas in the base
files: block-sparse uses 6 units/task; dense uses 6 units total; etc.

##### 3a. (random-sparse only) Pre-sweep to pick the per-task unit count

Before the main n_tasks sweep, run the sparsity × LR grid at n=32 to
identify the best per-task unit count for `random_sparse`:

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/03_nonstationary_n_tasks/sparsity_search/random_sparse.yaml
```

The winning `model.initial_hidden_units` value corresponds to a per-task
unit count {12, 24, 48, 96} = {384, 768, 1536, 3072} / 32. Update
`base_random_sparse.yaml` to bake that in:

- Set `model.initial_hidden_units` / `max_hidden_units` to
  `${eval:<k>*${task.n_tasks}}` with `<k>` = winning per-task count.
- Set `model.init_random_range_out: [20, 20]`.
- Set `model.init_random_range_in` to the formula-derived range at the
  matching budget.

##### 3b. Main n_tasks sweep

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/03_nonstationary_n_tasks/sweep/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/03_nonstationary_n_tasks/sweep/dense.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/03_nonstationary_n_tasks/sweep/random_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/03_nonstationary_n_tasks/sweep/structure_search.yaml
```

No `best/` follow-ups for (3) — the sweep itself reports the per-method,
per-n_tasks asymptotic accuracy/loss curves.
