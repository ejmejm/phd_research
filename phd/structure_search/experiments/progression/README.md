# Structure-Search Progression

A progression of experiments that stitches together what we've learned
about connectivity search into a single narrative:

1. **Static baselines** at a matched ~30k connection budget — dense vs
   block-sparse vs random-sparsity — to establish the oracle, the chance
   baseline, and the middle ground.
2. **Random feature search**: prune + generate whole hidden units at a
   fixed 30k budget, where each generated unit has random connectivity
   (`n_in ~ U[1, 128]`, `n_out ~ U[1, 20]`).
3. **Mixed feature search**: half of generated units stay within their
   task column; the other half are free (same random-sparsity as step 2).
4. **Connection-level feature search**: same feature-level generation as
   step 2, but prune individual weights (not whole units). A unit is
   pruned only when all its outgoing connections are zero.
5. **Pruned feature search** (deferred): empty-init + signed-LOO unit
   pruning + statistical-threshold weight pruning.

Builds on `../local_pruning_progression/` (weight-level dynamics on a
linear model) and `../column_guided_search.py` (feature-level dynamics
on a multi-layer sparse network).

## Metrics

All metrics are reported as mean ± 95% CI over 20 seeds unless otherwise
noted. Runs use `n_tasks=2` parallel MNIST (concatenated digits, 1568
input → 20 output), SGD, batch size 1, 225k training steps.

### Loss / accuracy
- **asymptotic_loss** — mean training cross-entropy loss over the last
  10% of the run (~22.5k steps). Measures rate of adaptation under
  non-stationarity (label permutations every 4000 steps), not peak
  accuracy. Lower is better.
- **asymptotic_accuracy** — per-task argmax accuracy averaged over the
  same window.
- **average_loss** — mean loss over the entire run (reference only).

### Structure / connectivity (logged by `metrics.compute_structure_metrics`)
- **layer_{l}/active_units** — number of active hidden units in layer
  `l` at the current log step (averaged across seeds).
- **layer_{l}/avg_incoming_conns** — mean fan-in per active hidden unit
  in layer `l`. Primary interest for step 4 (hypothesis: connection-
  level pruning collapses most units to near-zero fan-in).
- **layer_{l}/avg_outgoing_conns** — mean fan-out (to hidden or output)
  per active hidden unit in layer `l`.
- **total_active_connections** — total active connections at the log
  step (hidden-layer incoming + hidden-to-output), mean across seeds.
- **total_pruned / total_generated** — connections removed / added per
  window (sum over `log_freq` steps).
- **median_utility / layer_{l}/avg_utility** — per-layer unit-utility
  statistics from the structure tracker.

## Invariants (constant across every step)

- **Task**: non-stationary parallel MNIST, `n_tasks=2`. One randomly
  chosen task has its label map replaced with a fresh permutation every
  **4000 steps** (`dataset.permute_period=4000`).
- **Input dim**: 1568 (2 × 784, raw MNIST; step 5 will use normalized
  MNIST via `load_mnist_normalized`).
- **Output dim**: 20 (2 × 10). Per-task softmax CE.
- **Architecture depth**: 1 hidden layer.
- **Activation**: `leaky_relu`.
- **Optimizer**: SGD, no momentum, no weight decay.
- **Batch size**: 1 (online).
- **Run length**: 225,000 steps.
- **Seeds**: 20 (seeds 42..61), vmapped for throughput.
- **LR sweep**: powers-of-2 grid `2**[-13, -11, -9, -7, -5, -3]`;
  extended if the best LR is at an edge.
- **Parameter / connection budget**: 30,000. All step-1 baselines
  configured to land within 5% of this. Steps 2–4 use
  `structure_tracker.connection_budget=30000` as a hard cap.
- **MLflow project**: `structure_search_progression`. Each step has its
  own sweep name (`step1_static_baselines`, `step2_random_feature_search`,
  etc.) under this project.
- **Sweep framework**: mlflow-sweeper (Python-inline `SweepConfig` +
  `run_sweep`, same pattern as `../local_pruning_progression/07_*.py`).
- **Initial connectivity of the dynamic network (steps 2–4)**: random-
  sparsity — each hidden unit at init samples `n_in ~ U[1, 128]` and
  `n_out ~ U[1, 20]` (same distribution as step 1's random-sparsity
  baseline). Implemented by `train._apply_random_sparsity` with
  `max_in=128`, `max_out=20`. New units generated during training
  sample from the same distribution (via `free_generate` with
  `random_input_count=True`).

## Step 1 — Static baselines

**Goal**: establish the oracle (block-sparse, task-aligned sub-MLPs),
the mediocre (random sparsity), and the chance baseline (dense) at a
matched 30k connection budget. Expected ordering on asymptotic loss:
**block_sparse > random_sparsity > dense** (in terms of task
performance; smaller loss = better).

### Sizing (empirically tuned, all within 1% of 30k)

| Variant           | Model                                          | Sizing parameter           | Active params/conns |
|:------------------|:-----------------------------------------------|:---------------------------|---------------------|
| `dense`           | `BlockSparseMLP(n_tasks=1, hidden_dim=19, ...)` — i.e. a standard MLP on the 1568-dim input | `hidden_dim=19`            | 30,172 (+0.6%)      |
| `block_sparse`    | `BlockSparseMLP(n_tasks=2, hidden_dim=19, ...)` — one sub-MLP per task | `hidden_dim=19`            | 30,172 (+0.6%)      |
| `random_sparsity` | `DynamicNetwork` at `units_per_layer=405`, `max_connections_per_unit=128`, `max_fan_out=20`, post-processed by `_apply_random_sparsity` to sample `n_in ~ U[1,128]` and `n_out ~ U[1,20]` per unit | `units_per_layer=405`      | avg 30,181 over 5 seeds (+0.6%) |

Structure-tracker is disabled (`structure_tracker.enabled=false`) —
connectivity is frozen, only weights are trained.

### Replication

```bash
cd phd/structure_search
python experiments/progression/01_static_baselines.py
```

Sweeps 3 variants × 6 LRs × 20 seeds = 18 trials of 20 seeds each under
MLflow project `structure_search_progression`, sweep name
`step1_static_baselines`.

## Step 2 — Random Feature Search

**Goal**: starting from the same random-sparsity init as step 1, does
dynamic feature-level pruning + generation improve over the frozen
random-sparsity baseline? Each cycle (`prune_frequency=200` steps)
prunes the fraction `prune_rate=0.0001` of active units with lowest
`normalized_contribution_utility`, then refills to budget = 30,000
via `free_generate` with `random_input_count=True`.

Each new unit gets:
- `n_in ~ U[1, 128]` inputs (LeCun-uniform weights), and
- `n_out ~ U[1, 20]` random outputs (zero-init weights).

Expected to land between random-sparsity and dense on asymptotic loss:
the search can concentrate connectivity on useful pixels, but it
cannot yet bias toward task alignment.

### Replication

```bash
cd phd/structure_search
python experiments/progression/02_random_feature_search.py
```

## Step 3 — Mixed Feature Search

**Goal**: does pairing free feature search with a column-aligned
subpopulation lift loss over pure random search? Each cycle generates
half the new units via `column_generate` (within-task inputs and
outputs, deterministic 128 incoming) and half via
`free_generate_protected` with `random_input_count=True` (unrestricted
inputs, outgoing protected from cross-task tagged units).

Everything else matches step 2 (same budget, same prune rate, same
utility).

### Replication

```bash
cd phd/structure_search
python experiments/progression/03_mixed_feature_search.py
```

## Step 4 — Connection-Level Feature Search

**Goal**: same network and feature generation as step 2, but prune
**individual weights** instead of whole units.
`ConnectionConnectivityManager` drives phase-1 weight pruning by
`contribution_connection_utility` at `prune_rate=0.0001`, then phase-2
dead-unit detection removes any unit whose total outgoing connections
reach zero, then phase-3 generates new units to refill the budget via
`free_generate` with **`random_input_count=False`** — new units get a
deterministic 128 incoming connections so the input count doesn't
compound with connection-level pruning's natural fan-in collapse.

Hypothesis: connection-level pruning may drive most surviving units to
near-zero fan-in. The `layer_0/avg_incoming_conns` metric tracks this
directly per `log_freq` window — if it crashes toward 1–2, the
hypothesis is confirmed.

### Replication

```bash
cd phd/structure_search
python experiments/progression/04_lr_sweep.py   # 5 LRs × 5 seeds
python experiments/progression/04_main.py       # single 20-seed run at best LR
```

## Step 5 — Pruned Feature Search (DEFERRED)

**Goal**: empty-init network that grows itself via feature-level
signed-LOO utility pruning + statistical-threshold weight pruning.
More details are pending from the user before implementation.

**Planned architecture** (subject to change):

- **Init**: empty network (`init_strategy='empty'`, no units, no
  connections).
- **Unit generation**: per new unit, 128 random inputs with
  LeCun-uniform weights and **one** random output connection with
  zero-init weight. Mirrors `single_output_generate`.
- **Unit pruning**: fixed-rate by signed LOO utility (`loo_utility`).
  New units start with utility = median of active units
  (`median_utility_init`, already wired as default in
  `column_guided_search.py`).
- **Weight pruning**: statistical-confidence threshold from
  `../local_pruning_progression/` step 6 (`prune_statistical`). A new
  manager class extending `ConnectionConnectivityManager` is required
  to track per-connection age and bias-corrected `U_corr`.
- **Input normalization**: `load_mnist_normalized()` — same as
  local_pruning_progression step 6, so `σ_x = 1` in the τ_w formula.

Pending details from the user:
- Exact feature prune rate and cadence.
- Whether the rate applies always, or only above a target unit count.
- How statistical weight pruning interacts with dead-unit detection.
- Logging granularity for per-unit age / fan-in / utility.

The placeholder script `05_pruned_feature_search.py` raises
`NotImplementedError` until the spec is finalized.

## Scripts

Each step has a companion `_lr_sweep.py` / `_main.py` split: the LR
sweep scans the grid at **5 seeds** via mlflow-sweeper; the main run
is a single **20-seed** MLflow run at the winning LR (loaded from the
sweep's Optuna study, not a sweep trial).

| Script                           | Purpose                                                                    |
|:---------------------------------|:---------------------------------------------------------------------------|
| `common.py`                      | Shared constants, MLflow/Optuna URI helpers, DictConfig factories          |
| `01_lr_sweep.py`                 | Step 1 LR × variant sweep over static baselines                            |
| `01_main.py`                     | Step 1 main — three 20-seed runs at the best LR per variant                |
| `02_lr_sweep.py`                 | Step 2 LR sweep; `variant='no_column'`, unit-level pruning                 |
| `02_main.py`                     | Step 2 main — single 20-seed run at best LR                                |
| `03_lr_sweep.py`                 | Step 3 LR sweep; `variant='mixed_generation'`                              |
| `03_main.py`                     | Step 3 main — single 20-seed run at best LR                                |
| `04_lr_sweep.py`                 | Step 4 LR sweep; `tracker_mode='connection'`, `variant='no_column'`        |
| `04_main.py`                     | Step 4 main — single 20-seed run at best LR                                |
| `05_pruned_feature_search.py`    | Step 5 stub — raises `NotImplementedError`                                 |

## Supporting library changes (backwards-compatible)

The new progression relies on a few small additions to the shared
library. All are opt-in — existing sweep configs produce bit-identical
results on a fixed seed:

- `free_generate(random_input_count=False)` and
  `free_generate_protected(random_input_count=False)` in
  `column_guided_search.py`: when True, each new unit samples
  `n_conns ~ U[1, half_conns]` instead of the fixed `half_conns`. When
  False (default), behavior is unchanged.
- `mixed_generate(random_input_count=False)`: threads the flag into the
  free half only.
- `train._apply_random_sparsity(model, key, max_in, max_out)`:
  post-processes a freshly-initialised `DynamicNetwork` to subsample
  per-unit input and output counts. Activated by
  `cfg.model.random_sparsity_init=True` in both `train.py` and
  `column_guided_search.py`'s `prepare_experiment`.
- `train.run_config(cfg)` and
  `column_guided_search.run_config(cfg)`: callable variants of the
  `@hydra.main` decorated `main` functions. They do **not** call
  `init_experiment` / `finish_experiment`, so mlflow-sweeper can
  drive them under a trial run it already started. `main()` keeps the
  CLI behavior by wrapping `run_config` with the MLflow lifecycle.
