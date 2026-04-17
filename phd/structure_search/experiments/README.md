# One-Off Experiments

This directory contains standalone training scripts for targeted experiments that
are narrower in scope than `train.py`. Each file is self-contained and runs its
own simplified training loop.

---

## Column-Guided Structure Search (`column_guided_search.py`)

### Motivation

Dynamic network experiments with greedy structure search (prune + generate) have  
failed to match the block-sparse MLP baseline at equal parameter budget. There  
are two competing explanations for why:

1. **Search quality**: The utility function and generation strategy are too noisy
  to reliably discover the optimal block-sparse connectivity.
2. **Loss of plasticity**: Even if the right structure were found, continual
  pruning and regeneration may degrade the network's ability to learn, causing
   it to underperform a baseline that never restructures.

These two failure modes are entangled in any realistic structure search run.
This experiment isolates (2) by using a guided structure search that is
**guaranteed to converge to independent subnetworks**, removing search quality
as a confound.

If the guided network matches block-sparse performance → plasticity is fine and
the search algorithm is the bottleneck.

If the guided network still underperforms → loss of plasticity is itself the
problem, regardless of whether the algorithm finds the right structure.

### How It Works

**Column assignment.** With `n_tasks` independent MNIST subtasks, the network
is divided into `n_tasks` vertical "columns". Each position in the network has
a deterministic column assignment:

- Input index `j` → column `j // (input_dim // n_tasks)` (MNIST pixels for task 0, task 1, ...)
- Hidden unit at layer position `i` → column `i // (max_units_per_layer // n_tasks)`
- Output index `j` → column `j // (output_dim // n_tasks)` (class logits for task 0, task 1, ...)

**Column utility (two-phase).** After each training step, every active unit is
assigned a utility score that drives which units get pruned:

- Count the unit's cross-column connections (incoming weight connections to
sources outside its column, plus output connections to outputs outside its
column).
- If `cross_count > 0`: `utility = -cross_count`. This is always negative,
meaning any unit with cross-column connections will be pruned before any
fully within-column unit, regardless of how useful it is for the loss.
- If `cross_count == 0`: `utility = contribution_utility` (the standard
activation × outgoing-weight magnitude measure). Once the network has
converged structurally, normal pruning pressure applies.

**Column generation.** When the pruner removes cross-column units and frees up
connection budget, the generator refills slots with new units that are
structurally constrained from birth:

- A unit in slot `(l, i)` is assigned to column `c = i // col_size`.
- Its incoming connections are sampled only from buffer positions in column `c`
(inputs in `[c×input_col_size, (c+1)×input_col_size)` and prior-layer hidden
units in slot positions `[c×col_size, (c+1)×col_size)`).
- Its output connections are restricted to output indices in
`[c×out_col_size, (c+1)×out_col_size)`.
- Each new unit draws `max_connections_per_unit // 2 = 32` incoming connections
and connects to all `output_dim // n_tasks = 10` outputs in its column.
- Incoming weights are initialised uniform in `[-1/√n_conns, 1/√n_conns]`;
output weights are initialised to zero and grown by the optimizer.

Over successive prune/generate cycles the network converges to fully independent
subnetworks — one per task — without any cross-task connections.

### Sweep (`conf/sweeps/dynamic_network/column_guided_stationary.yaml`)

Mirrors the `small_dynamic_stationary` sweep but replaces the random utility and
generation with the column-guided variants. Key configuration choices:

- `**model.hidden_dim: 520`** — sets the number of active units per layer at
initialisation. Each unit draws 32 incoming connections and the last layer
draws 32 sparse output connections, giving an initial connection count of
`520 × (32 + 32 + 32) = 49,920 ≈ 50k`. Both 520 and `max_units_per_layer`
are divisible by `n_tasks=5` so each column starts with an equal 104 units.
- `**model.max_units_per_layer: 1040**` — 2× `hidden_dim`; the remaining 520
empty slots per layer serve as the reservoir that `column_generate` fills.
- `**model.max_connections_per_unit: 64**` — hard cap on incoming connections;
unchanged from the standard dynamic config so there is room to grow after
cross-column output connections are replaced by within-column hidden ones.
- `**structure_tracker.connection_budget: 50_000**` — matches
`block_sparse_stationary` so the total parameter count is comparable.

**Sparse output initialisation.** `init_random_dynamic_network` wires every
last-layer unit to all `output_dim` outputs by default. A post-init step
(`_sparsify_outputs`) replaces this with 32 randomly selected outputs per unit
— the same number that `column_generate` uses — so the network starts in a
regime that mirrors what will be maintained throughout training.

The sweep crosses learning rate × prune rate to find the best configuration for
comparison against the block-sparse baseline.