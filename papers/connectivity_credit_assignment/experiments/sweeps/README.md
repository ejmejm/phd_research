# Running the sweeps

Every command runs from `experiments/`. Each sweep is a Comet Optimizer grid;
`comet_sweep` registers it and prints a sweep id, and agents then pull trials
from that id until the grid is exhausted, so the same command scales from one
local worker to a cluster array.

```bash
cd experiments
comet_sweep -c sweeps/<path>.yaml       # register the grid; prints a sweep id
comet_sweep -s <sweep-id>               # run trials against it
comet_sweep -s <sweep-id> -n 4          # ... stopping after 4 trials
```

Set `COMET_API_KEY` and `COMET_WORKSPACE` in your environment. The Comet
project each sweep logs to is the `project:` field at the top of its config;
change it and your runs land somewhere other than the published projects.

## The order things run in

Two of the three families need two passes. The step-size sweep is not a
formality — the central claim of Figure 3 is about which step-sizes each
connectivity pattern can tolerate, so the grid has to bracket each method's
optimum rather than be inherited from another method.

### Figures 1 and 2 — matched connectivity, and dense scaling

**Pass 1, step-size sweeps** (5 seeds per cell, enough to pick a step-size):

```bash
comet_sweep -c sweeps/01_matched_and_scaling/sweep/block_sparse_stationary.yaml
comet_sweep -c sweeps/01_matched_and_scaling/sweep/dense_stationary.yaml
comet_sweep -c sweeps/01_matched_and_scaling/sweep/block_sparse_nonstationary.yaml
comet_sweep -c sweeps/01_matched_and_scaling/sweep/dense_nonstationary.yaml
```

**Pass 2**: read the winning step-size per cell out of
`analysis/01_matched_and_scaling.ipynb` (the last section plots the
sensitivity curves), write it into the matching `best/` config, then run the
30-seed versions that produce the figures:

```bash
comet_sweep -c sweeps/01_matched_and_scaling/best/block_sparse_stationary.yaml
comet_sweep -c sweeps/01_matched_and_scaling/best/dense_stationary.yaml
comet_sweep -c sweeps/01_matched_and_scaling/best/block_sparse_nonstationary.yaml
comet_sweep -c sweeps/01_matched_and_scaling/best/dense_nonstationary.yaml
```

The dense `best/` configs sweep seven budgets whose best step-sizes differ, so
they use the `switch` resolver to map each budget to its own winner rather
than splitting into seven files:

```yaml
optimizer.learning_rate: "${eval:'2**${switch:${model.initial_hidden_units}, 6, -9, 12, -9, 24, -9, 48, -9, 96, -9, 192, -10, 384, -10}'}"
```

Keys match as strings, so list integers unquoted. The `eval` body needs the
single quotes — Hydra's override parser otherwise chokes on the commas inside
the nested `${switch:...}`.

### Figure 3 — the dense transition

One pass; the method axis and the step-size axis sweep together.

```bash
comet_sweep -c sweeps/02_dense_transition/sweep/main.yaml
```

The three methods are three values of `algorithm.transition_step`: `0` is
dense from the start, `250000` is the transition, and `600000` exceeds
`train.total_steps` so the fill never fires — that is the block-sparse
baseline, running through exactly the same code path. The sentinel exists
because Comet's Optimizer rejects `null` inside a numeric list; the trainer
treats any step past `total_steps` as "never", and the two are verified to
give identical results.

### Figure 4 — scaling the problem and the network

One pass, and no `best/` stage: the sweep itself is the figure, with each cell
reported at its own best step-size.

```bash
comet_sweep -c sweeps/03_task_scaling/sweep/block_sparse.yaml
comet_sweep -c sweeps/03_task_scaling/sweep/dense.yaml
```

Block-sparse cells with fewer hidden units than tasks fail fast by design — it
needs at least one unit per task. That is 12 of the 40 cells, and the analysis
notebook shows them as gaps.

## Seeds, and why 30 of them are split in half

The `best/` configs and the Figure 3 sweep run 30 seeds per configuration. One
trial vmaps its seeds, so 30 at the 384-unit budget does not fit a 10 GB MIG
slice. `seed_offset` splits them into two 15-seed halves that run as separate
trials; `stack_per_seed` in the analysis rejoins them along the seed axis.

Figures 1 and 3 report 95% confidence intervals over those 30 seeds. They are
narrower than the line width, which is worth stating rather than hiding.

## Running on a cluster

An agent is one process pulling trials from a sweep id, so an array job is
just N of them against the same id:

```bash
sbatch --array=1-10 --gpus=1 --cpus-per-task=1 --mem=10G --time=03:00:00 \
  launch_comet_agent.sbatch -s <sweep-id> -p <repo-path>/experiments
```

`launch_comet_agent.sbatch` is site-specific and not included; it needs to
activate your environment, `cd` to the repo's `experiments/` directory, and
run `comet_sweep -s "$SWEEP_ID"`.

Grid sizes, so you can budget: Figure 1/2 step-size sweeps are 6 + 42 + 4 + 28
= 80 trials; the `best/` stage is 2 + 14 + 2 + 14 = 32 trials of 15 vmapped
seeds each; Figure 3 is 30 trials plus a step-size addendum; Figure 4 is 160
trials per method. Every trial is 500,000 single-sample steps.

## Divergence is a result

A diverged run is a successful measurement of a bad step-size. The trainer
detects it, stops early, and records a finite sentinel with `diverged=1` —
because a sweep backend cannot store `inf` as an objective, and a trial that
records nothing gets re-assigned and recomputes the same divergence until it
hits `retryAssignLimit`.

Downstream, the analysis keys on the `diverged` flag rather than the sentinel:
a diverged cell becomes NaN and shows as a gap. That is why the dense and
dense-transition lines in Figure 3's sensitivity panel stop instead of
plunging to zero.
