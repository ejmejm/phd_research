## Replication

Four experiments at 4× the per-task parameter budget of
`weight_pruning_lr_sweep` (so block-sparse and dense both target ~19056
total weights at n_tasks=4). All commands run from `phd/structure_search/`.

The general workflow for each numbered experiment is:

1. Run the LR sweeps under `sweep/`.
2. Pick the best LR for each method/budget from the sweep results.
3. Fill in the `???` placeholder under `best/<method>.yaml`'s
   `optimizer.learning_rate` field with that LR.
4. Run the fixed-LR follow-up under `best/` (30 seeds, 101–130).

#### Per-value best LRs via the `switch` resolver

When a `best/` config sweeps over an inner axis (e.g. dense
`initial_hidden_units` in (1)/(2), or `task.n_tasks` in (3),
`task.permute_period` in (4)) and different values of that axis have
different best LRs, **do not split the config into one file per value**.
Use the `switch` resolver instead — it's registered in
`phd/feature_search/jax_core/experiment_helpers.py` and looks up a value
by exact-string match against a (key, value, key, value, …) list:

```yaml
parameters:
  model.initial_hidden_units: [6, 12, 24, 48, 96, 192, 384]
  optimizer.learning_rate: "${eval:2**${switch:${model.initial_hidden_units}, 6, -11, 12, -10, 24, -10, 48, -9, 96, -9, 192, -8, 384, -8}}"
```

Each grid combination picks the matching LR for that budget. Keys are
matched as strings, so list scalar integers/floats without quotes; if a
value has spaces, wrap the whole interpolation in quotes. This keeps one
`best/<method>.yaml` per method instead of one per budget.

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

### 4. Nonstationary, scaling permute period: all four methods

Same nonstationary 4-task setup as (2), but sweeps `task.permute_period ∈
{200, 400, 800, 1600, 3200, 6400}` to vary the level of non-stationarity
while holding `n_tasks=4` and per-task capacity constant. Base configs
match (2)'s n=4 resolution: block-sparse uses 24 units; dense uses 6
units; structure-search uses max_hidden=800, budget=19000, prune=256,
units_per_event=8.

For `random_sparse`, reuse the per-task unit count that won
`02_nonstationary/sweep/random_sparse.yaml` — there is no separate
sparsity search in (4). Update `base_random_sparse.yaml` accordingly:

- Set `model.initial_hidden_units` / `max_hidden_units` to the winning
  total unit count (one of {48, 96, 192, 384}).
- Set `model.init_random_range_in` to
  `[1, int(2 * (19056 / <units> - 20) - 1)]`.
- Confirm `model.max_connections_per_unit` ≥ the new `range_in` upper.

```bash
mlflow-sweep conf/sweeps/weight_pruning_scaling/04_nonstationary_level/sweep/block_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/04_nonstationary_level/sweep/dense.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/04_nonstationary_level/sweep/random_sparse.yaml
mlflow-sweep conf/sweeps/weight_pruning_scaling/04_nonstationary_level/sweep/structure_search.yaml
```

No `best/` follow-ups for (4) — the sweep itself reports the per-method,
per-permute-period asymptotic accuracy/loss curves.
