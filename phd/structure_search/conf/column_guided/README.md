# Column-Guided Structure Search

## Overview

This directory contains experiments investigating whether a dynamic sparse
network can learn to discover task-aligned structure (independent per-task
subnetworks) through greedy pruning and generation on a parallel multi-MNIST
task with 5 independent classification subtasks.

The experiments progress through a series of questions, each building on the
findings of the previous one.

---

## Phase 1: Can guided structure search match block-sparse performance?

**Motivation.** Greedy structure search (prune low-utility units, generate random
replacements) has failed to match a hand-designed block-sparse baseline.  Two
possible explanations: (1) the search algorithm can't find good structure, or
(2) continual pruning/regeneration causes loss of plasticity.

**Approach.** Use a "column-guided" search that is *guaranteed* to converge to
independent subnetworks (one per task), removing search quality as a confound.
If guided search matches block-sparse → plasticity is fine and the search
algorithm is the bottleneck.

**Config:** `stationary` (the baseline)

```bash
python experiments/column_guided_search.py --config-name stationary
```

**Result.** Guided search reaches ~95.6% test accuracy, matching block-sparse.
Plasticity is not the problem — the search algorithm is.

### Incremental relaxation

To isolate which component of the guided search matters, three configs
progressively remove the column guidance:


| Config            | Utility fn                        | Generation fn             | Test Acc |
| ----------------- | --------------------------------- | ------------------------- | -------- |
| `stationary`      | `column_utility`                  | `column_generate`         | ~95.6%   |
| `relaxed_outputs` | `column_utility`                  | `column_generate_relaxed` | ~95.6%   |
| `normal_utility`  | `normalized_contribution_utility` | `column_generate_relaxed` | ~89.4%   |
| `no_column`       | `normalized_contribution_utility` | `free_generate`           | ~89.4%   |


**Conclusion.** The column-based utility function (not the generation strategy)
is the dominant factor.  Without it, performance drops to ~89% regardless of
how units are generated.

---

## Phase 2: Can a general utility function discover column structure?

The column-based utility function explicitly penalizes cross-column connections.
The question is whether a *general-purpose* utility function (normalized
contribution utility) would naturally favor within-column units, which would
mean greedy pruning could discover the right structure without being told what
it is.

### Experiment: Utility Comparison (`utility_comparison`)

**Question.** Is the average utility of units in a column-constrained network
higher than in a network with random cross-column connections?

Two runs, both with no pruning or generation — just training and utility
tracking:

- `init_mode=random`: Normal random init (cross-column connections)
- `init_mode=column`: All connections within-column; 104 units per column per
layer (evenly distributed)

```bash
python experiments/column_guided_search.py --config-name utility_comparison init_mode=random
python experiments/column_guided_search.py --config-name utility_comparison init_mode=column
```

**Result (30k steps).** Column-init reaches 91.8% vs 87.0% for random-init,
confirming that within-column structure helps.  However, this does not tell us
whether the utility function *favors* within-column units — just that the
network performs better with that structure.

### Experiment: Mixed Generation (`mixed_generation`)

**Question.** If we generate both within-column and cross-column units side by
side, will normalized contribution utility preferentially keep within-column
units?

Half of generated units use `column_generate` (tagged `column_tag=1`), the
other half use `free_generate_protected` (tagged `column_tag=2`).  Initial
units from the random init are tagged `column_tag=0`.  Free units are prevented
from pushing cross-column outgoing connections to tagged units, so column
units stay purely within-column.

```bash
python experiments/column_guided_search.py --config-name mixed_generation
```

Per-unit snapshots are captured every training step and saved as
`unit_snapshots.npz`.  Analysis:

```bash
python experiments/plot_mixed_generation.py --run-id <RUN_ID>
```

**Result (30k steps, stationary).** Neither column nor free generated units are
adopted.  Both have ~4% survival rate vs 54% for initial units.  The problem is
**blocking**: initial units establish high utility early and prevent any new
unit — regardless of connectivity — from competing.

---

## Phase 3: Non-stationarity as a solution to blocking

**Key insight.** In the stationary setting, initial units monopolize utility and
block new units from being adopted.  But when the task changes (label
permutations), error spikes, existing units' utility drops, and newly generated
features get a chance to prove themselves.  If within-column features are
genuinely better structured, they should be preferentially adopted during these
transition windows.

### Experiment: Non-stationary sweep (`mixed_generation_ns_sweep`)

Sweep over `permute_period` (how often a random task's labels are permuted) to
test whether non-stationarity unblocks feature adoption.

```bash
mlflow-sweeper conf/column_guided/mixed_generation_ns_sweep.yaml
```

**Result.** Higher non-stationarity → more new features adopted.  The surviving
new features are overwhelmingly column-constrained.  This confirms that:

1. Non-stationarity solves the blocking problem
2. When blocking is removed, within-column features *are* preferentially kept
3. The utility function can discover structure — it just needs churn to overcome
  the incumbency advantage of established features

### What's next: does this actually help performance?

The NS sweep showed that non-stationarity enables structural learning, but the
question remains: does the learned structure actually improve performance
compared to random connectivity?

**Hypothesis.** On the non-stationary problem, a network that adopts
within-column structure via mixed generation should outperform a network with
fixed random connectivity (which serves as a lower baseline).  A network
initialized with within-column connectivity serves as the upper baseline.

The `permute_stop` parameter allows freezing the task after a period of
non-stationarity, so the network converges to a final stationary accuracy with
whatever structure it has learned.

### Experiment: NS Utility Comparison (`ns_utility_comparison`)

Upper and lower baselines for non-stationary performance — same as the
stationary utility comparison but with `permute_period=2000`:

```bash
# Lower baseline: random init, no structure search
python experiments/column_guided_search.py --config-name ns_utility_comparison init_mode=random

# Upper baseline: column init, no structure search
python experiments/column_guided_search.py --config-name ns_utility_comparison init_mode=column
```

### Experiment: NS Mixed Generation (`ns_mixed_generation`)

Mixed generation with non-stationarity — does the network learn structure that
closes the gap between the random and column baselines?

```bash
python experiments/column_guided_search.py --config-name ns_mixed_generation
```

---

## Reference: Network Architecture

`init_random_dynamic_network` creates the initial network with `hidden_dim=520`
active units per layer (out of `max_units=1040`), `max_connections_per_unit=64`,
`max_fan_out=64`, 2 hidden layers.

**Column layout.** `col_size = max_units // n_tasks = 208`.  With 520 initial
units in slots 0-519: Col 0 gets 208, Col 1 gets 208, Col 2 gets 104,
Cols 3-4 are empty.  When `init_mode=column`, units are distributed evenly
(104 per column).

**Column assignment** is purely positional:

- Input pixel `p` → column `p // (input_dim // n_tasks)`
- Hidden unit slot `u` → column `u // (max_units // n_tasks)`
- Output neuron `k` → column `k // (output_dim // n_tasks)`

`**column_utility`** (used in guided search): if a unit has any cross-column
connections, `utility = -cross_count` (pruned first); otherwise
`utility = contribution_utility`.

`**normalized_contribution_utility`** (used in mixed generation): `mean|act| × sum|outgoing_weights| / max(n_outgoing, 1)`.  No column bias.

**Structure modification** happens every `prune_frequency=200` steps.  The prune
accumulator grows by `prune_rate × active_connections` each step and carries
over unused budget across cycles.  Generation fills up to
`connection_budget=50,000` total connections.

## Reference: Config Options


| Option                      | Default         | Description                                     |
| --------------------------- | --------------- | ----------------------------------------------- |
| `variant`                   | `column_guided` | Which utility/generation functions to use       |
| `init_mode`                 | `random`        | `random` or `column` (within-column init)       |
| `dataset.permute_period`    | `0`             | Steps between label permutations (0=stationary) |
| `dataset.permute_stop`      | `0`             | Stop permuting after this step (0=never)        |
| `structure_tracker.enabled` | `true`          | Set `false` to disable pruning/generation       |
| `snapshot_subsample`        | `10`            | Keep every Nth step in snapshot data            |


