# Column-Guided Structure Search

## Motivation

Dynamic network experiments with greedy structure search (prune + generate) have
failed to match the block-sparse MLP baseline at equal parameter budget. There are
two competing explanations:

1. **Search quality**: The utility function and generation strategy are too noisy
   to reliably discover the optimal block-sparse connectivity.
2. **Loss of plasticity**: Even if the right structure were found, continual pruning
   and regeneration may degrade the network's ability to learn.

These two failure modes are entangled in any realistic structure search run. This
experiment isolates (2) by using a guided structure search that is **guaranteed to
converge to independent subnetworks**, removing search quality as a confound.

If the guided network matches block-sparse performance → plasticity is fine and the
search algorithm is the bottleneck.

If the guided network still underperforms → loss of plasticity is itself the problem.

## Run

From `phd/structure_search/`:

```bash
python experiments/column_guided_search.py --config-dir conf/column_guided --config-name stationary
```

## Network Initialization

`init_random_dynamic_network` creates the initial network with `hidden_dim = 520`
active units per layer (out of a buffer of `max_units = 1040`).

**Column layout.** Units are placed sequentially into slots 0–519 per layer. With
`col_size = max_units // n_tasks = 208`:

- Col 0: slots 0–207 (208 active)
- Col 1: slots 208–415 (208 active)
- Col 2: slots 416–519 (104 active), 520–623 inactive
- Col 3 & 4: entirely inactive

**Incoming connections.** For each layer `l`, each of the 520 active units draws
`max_connections_per_unit = 64` sources randomly from the pool of all inputs + all
active units in layers 0..l-1 (no column restriction). Weights are LeCun uniform
scaled by `1/√n_conns`.

**Output connections** (`connect_all_to_output = False`): only the last hidden layer
(layer 1) is wired to all `output_dim = 50` outputs. Weights are LeCun uniform.

**`_sparsify_outputs`** (called immediately after). Replaces the dense all-to-50
output wiring with `max_fan_out // 2 = 32` randomly selected outputs per last-layer
unit. Selection is uniform random with no column restriction — so each unit starts
with ~22 cross-column output connections.

**Utility EMA init.** All units start with `unit_stats.utility = 0`, `age = 0`.

## New Unit Creation

Triggered every `prune_frequency = 200` steps when active connections are below
`connection_budget = 50,000`. Each prune cycle runs `column_generate` followed by
`column_assign_outgoing`.

### Slot selection

All `max_layers × max_units = 2080` slots are shuffled via uniform noise + argsort;
inactive slots float to the front. The first `max_new_units_per_step = 256`
candidates are taken. Each candidate has a `(layer, unit_position)` and an implied
column `col = unit_position // col_size`.

### Incoming connections (`column_generate`)

For each candidate at `(layer, col)`, the available source pool is:

- Input pixels in `[col × input_col_size, (col+1) × input_col_size)` — exactly that
  task's MNIST image (784 pixels)
- Active hidden units in layers `< layer`, same column

Up to `max_conns // 2 = 32` sources are drawn uniformly from this pool. Weights are
uniform in `[-1/√n_conns, 1/√n_conns]`. Candidates with no available same-column
sources are skipped. A cumulative budget check culls the list: cost per unit is
`n_incoming + max_out` connections.

Structural fields written: `input_indices`, `weights`, `unit_mask = 1`,
`activation_indices = 0`. EMA utility is set to `init_utility` (median of active
units); age is set to 0.

### Outgoing connections (`column_assign_outgoing`)

Called immediately after structural fields are set, with the already-updated
`unit_mask`. For each generated unit, a downstream pool is built:

- **Output targets:** the `out_col_size = 10` output neurons for that task/column
- **Hidden targets:** active units in later layers, same column, with a free
  `input_indices` slot

The pool is shuffled and up to `max_out = 32` targets are selected. A target is
only connected if it is actually in the available pool — this prevents filling spare
slots with cross-column targets when fewer than 32 within-column targets exist (e.g.
layer-1 units, which have 0 downstream hidden units, get at most 10 output
connections). For an output target, `output_mask[k, source_bp] = 1` and
`output_weights[k, source_bp] = 0`. For a hidden target, `source_bp` is scattered
into the first empty slot of that unit's `input_indices` with weight = 0.

**Net result:** a new unit has up to 32 within-column incoming connections with
random weights, and up to 32 within-column outgoing connections (output or
pushed-hidden) initialized to zero and grown by the optimizer.

## Column Assignment and Utility

The column of each unit/source is determined entirely by position:

| Source type | Column formula |
|-------------|---------------|
| Input pixel at index `p` | `p // (input_dim // n_tasks)` |
| Hidden unit at layer-buffer slot `u` | `u // (max_units // n_tasks)` |
| Output neuron at index `k` | `k // (output_dim // n_tasks)` |

`column_utility` assigns utility as follows:

- Count all incoming connections (`input_indices >= 0`) whose source column differs
  from the unit's column, plus all output connections (`output_mask = 1`) to
  out-of-column output neurons. Call this `cross_count`.
- If `cross_count > 0`: `utility = -cross_count` (always prune before any
  within-column unit).
- If `cross_count == 0`: `utility = contribution_utility` = `mean|activation| ×
  Σ|outgoing weights|`.

Over successive prune/generate cycles the network converges to fully independent
subnetworks — one per task.

## Incremental Relaxation Experiments

Three additional configs relax the column guidance incrementally to isolate
which component (utility function vs generation restriction) drives performance:

| Config | Utility | Generation | Result |
|--------|---------|-----------|--------|
| `stationary` (baseline) | `column_utility` | `column_generate` | ~95.6% |
| `relaxed_outputs` | `column_utility` | `column_generate_relaxed` | ~95.6% |
| `normal_utility` | `normalized_contribution_utility` | `column_generate_relaxed` | ~89.4% |
| `no_column` | `normalized_contribution_utility` | `free_generate` | ~89.4% |

**Conclusion**: The column-based utility function (not the generation restriction)
is the dominant factor.  Without it, performance drops regardless of how units
are generated.

## Experiment: Utility Comparison (`utility_comparison`)

**Question**: Is the average/sum utility of units in a column-constrained network
higher than in a random network?

Two runs with the same architecture, both with **no pruning or generation** — just
training and utility tracking via `normalized_contribution_utility`:

- **Run A** (`init_mode=random`): Normal random init (cross-column connections).
- **Run B** (`init_mode=column`): All connections within-column; units evenly
  distributed across columns (104 per column per layer).

At end of training, plotly histograms of per-unit utility distributions are logged
as mlflow artifacts (one per seed per layer), plus scalar summary metrics.

```bash
# Run A
python experiments/column_guided_search.py --config-name utility_comparison init_mode=random
# Run B
python experiments/column_guided_search.py --config-name utility_comparison init_mode=column
```

## Experiment: Mixed Generation (`mixed_generation`)

**Questions**:
1. If units are generated with and without column constraints, do the
   within-column units develop higher utility?
2. How frequently do within-column units survive pruning vs free units?

Uses `normalized_contribution_utility` (no column bias in utility).  Half of
generated units use `column_generate` (column-constrained incoming + outgoing)
and are tagged `column_tag=1`.  The other half use `free_generate_protected`
(unrestricted incoming, outgoing restricted from targeting tagged units in other
columns).  This prevents free units from contaminating tagged units with
cross-column connections.

Per-unit snapshots (utility, mask, age, tag) are captured at every training step
inside JIT and subsampled outside for storage.  The full snapshot data is saved
as `unit_snapshots.npz` mlflow artifact.

```bash
python experiments/column_guided_search.py --config-name mixed_generation
```

### Analysis

```bash
python experiments/plot_mixed_generation.py --run-id <RUN_ID> --fraction 0.1
```

Produces per-layer and global interactive plotly plots of feature utility
trajectories (blue = column-constrained, red = free), with mean/median/threshold
overlays.  Also prints summary statistics (survival rates, mean lifetime,
utility comparisons) to answer Q2 and Q3.
