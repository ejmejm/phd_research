# 04 — connectivity-learning methods

16-task non-stationary multi-MNIST, one task re-permuted every 125 steps —
the Figure 3 setting. Figure 3 compared two *fixed* connectivity patterns;
this family adds three methods that *change* connectivity during training, so
the comparison is against algorithms rather than against architectures.

| Arm | `model.init_strategy` | `algorithm.name` | Connectivity |
|---|---|---|---|
| dense | `dense` | `static` | everything connected |
| block-sparse | `block_sparse` | `static` | the known ideal |
| static sparse | `sparse` | `static` | SET's ER init, frozen |
| SET | `sparse` | `set` | magnitude prune, random regrow |
| DEEP-R | `sparse` | `deep_r` | L1 + Langevin, sign-flip prune |

Every arm runs on `padded_mlp`, so connectivity is a mask over a dense matrix
and the five arms differ in nothing but their masks. Two consequences matter:
the connection budget is preserved **per layer**, which is what both the SET
and DEEP-R papers specify and what the per-unit slot storage of
`dynamic_network` cannot express; and `path_purity` is computable, so you can
ask directly whether a method rediscovers the block-sparse structure rather
than only whether it performs well.

The cost is that a masked dense matmul costs the dense amount however sparse
the mask. That is affordable at 16 tasks and is the reason `dynamic_network`
exists; setting `model.type: dynamic_network` transparently selects the sparse
implementations in `algorithms/dynamic/` instead.

The static-sparse arm is the control that separates *sparse* from *learning
the sparsity*: identical budget, identical degree distribution, identical
representation, evolution removed.

## What is held constant

**The weight budget: 203,264 connections** — the block-sparse reference's
16 units/task × 794. Every arm but dense sits at exactly this number. Dense is
16× larger at the same width, as in Figure 3, because that *is* the dense
condition.

`base.yaml` derives the budget from `block_sparse_units_per_task` so the tie
to block-sparse is in the config rather than in a comment.

## Width is the sparsity axis

At a fixed budget, hidden width decides how thinly the budget is spread:

| `target_hidden_units` | units/task | W1 weights/unit | ε |
|---|---|---|---|
| 256 | 16 | 794 | 15.38 |
| 512 | 32 | 397 | 14.81 |
| 768 | 48 | 265 | 14.27 |
| 1024 | 64 | 198 | 13.78 |

H=256 is the interesting anchor: 794 weights per unit is *exactly* the
block-sparse profile (784 inputs + 10 outputs), only scattered across all 16
images instead of concentrated in one. Any gap there is attributable to where
the connections point, not how many there are.

Sweeping width at a fixed budget **is** sweeping SET's ε — the Erdős–Rényi
construction is untouched. `derive_sizes` runs the other way
(`H = floor((budget/ε − I − O)/2)`), so `base.yaml` inverts it. Grid on width
rather than ε because ε's usable range here is 13.8–15.4 — the paper's default
ε=20 is not even reachable at this budget, since `budget/20 < I+O` at 16 tasks.

## Order of operations

### 1. Pilot — how long is long enough

500,000 steps is the Figure 3 length, and the sparse methods probably need
more. Run one long, single-configuration probe per arm at the best guess and
look at where each curve flattens. Every command is run from the repo root.

```bash
COMMON="--config-path=sweeps/04_connectivity_methods --config-name=base \
  train.total_steps=2_000_000 comet_ml=true log_individual_seeds=true \
  project=paper-weight-pruning-connectivity-pilot seed=[0,1,2]"

# SET
python experiments/train.py $COMMON model.type=dynamic_network \
  algorithm.name=set algorithm.zeta=0.01 algorithm.evolve_frequency=125 \
  target_hidden_units=768 'optimizer.learning_rate=${eval:2**-7}'

# DEEP-R
python experiments/train.py $COMMON model.type=dynamic_network \
  algorithm.name=deep_r algorithm.l1=1e-3 algorithm.temperature=1e-5 \
  target_hidden_units=768 'optimizer.learning_rate=${eval:2**-7}'

# static sparse
python experiments/train.py $COMMON model.type=dynamic_network \
  algorithm.name=static target_hidden_units=768 \
  'optimizer.learning_rate=${eval:2**-7}'

# block-sparse and dense, to see whether they are still moving at 500k
python experiments/train.py $COMMON model.type=padded_mlp \
  model.init_strategy=block_sparse 'optimizer.learning_rate=${eval:2**-8}'
python experiments/train.py $COMMON model.type=padded_mlp \
  model.init_strategy=dense 'optimizer.learning_rate=${eval:2**-9}'
```

Then set `train.total_steps` in `base.yaml` to the longest arm's plateau and
leave it there for everything — the arms have to share a length to be
comparable.

### 2. Sweeps — 5 seeds per cell

```bash
comet_sweep -c sweeps/04_connectivity_methods/sweep/set.yaml
comet_sweep -c sweeps/04_connectivity_methods/sweep/static_sparse.yaml
comet_sweep -c sweeps/04_connectivity_methods/sweep/deep_r.yaml
```

Only run `sweep/block_sparse.yaml` and `sweep/dense.yaml` if the pilot pushed
`total_steps` past 500,000. At 500,000 the published Figure 3 runs already
cover both arms and should be reused.

| Sweep | Grid | Trials |
|---|---|---|
| `set` | 4 width × 4 ζ × 3 evolve_freq × 4 LR | 192 |
| `static_sparse` | 4 width × 4 LR | 16 |
| `deep_r` | 3 width × 3 l1 × 3 T × 4 LR | 108 |

Throughput at H=768 with 5 vmapped seeds, measured on an RTX 4080: 29 s per
20k steps for static sparse, 61 s for DEEP-R at the published `event_period:
1`, 90 s for SET at `evolve_frequency: 125`. SET is the expensive arm here,
because its prune is an order statistic over the full matrix and so costs a
sort that DEEP-R avoids entirely.
| `block_sparse` | 5 LR | 5 |
| `dense` | 5 LR | 5 |

### 3. Finals — 30 seeds at each arm's best cell

Not written yet; they follow the `best/` pattern of `01_matched_and_scaling`,
with `seed_offset` splitting 30 seeds into two 15-seed halves.

## Things that will bite

**`evolve_frequency` and `event_period` must divide `train.log_freq`**, which
is asserted in `run_experiment`. At `log_freq: 1000` that is the divisors of
1000, so 125/500/1000 are available and 2000 — one full drift cycle per task —
is not without raising `log_freq`.

**ζ and `evolve_frequency` are partly redundant.** What mostly matters is the
turnover rate ζ/`evolve_frequency`, which spans 1e-6 to 8e-4 per step across
this grid with duplicates in the middle. They are kept as separate axes
because rare large prunes and frequent small ones are not the same thing
against a 2000-step drift period, but if 192 trials is too many, collapsing
them to a single rate axis is the first cut to make.

**`event_period` does not change DEEP-R's dynamics.** Zero-crossings are
detected on every step; `event_period` only sets how long a connection stays
dormant before a random replacement is drawn. It is a cost/fidelity dial, not
a change to the sampler.

**DEEP-R contains no sort.** Its pruning is a sign test and its reactivation
is a uniform draw over dormant connections, so neither needs an ordering. The
draw is done as independent Bernoulli selection at `deficit / n_dormant`,
which lands the layer count within a fraction of a percent of target rather
than exactly on it, and self-corrects because the deficit is measured against
the absolute target every event. The exact-size alternative costs a sort of
the whole matrix -- 521 ms against 15 ms per event at H=768 -- and buys
nothing here. SET does sort, because "the smallest-magnitude zeta fraction" is
an order statistic and there is no way around it.

**DEEP-R's defaults do nothing at this scale.** `l1=1e-5, T=1e-7` from the
code's signature move a typical weight by ~1% of its magnitude over a whole
drift period, so DEEP-R would silently reduce to the static-sparse arm.
`sweep/deep_r.yaml` brackets `l1=1e-3` and `T=1e-5` instead, which are the
values that walk a typical weight to zero in ~2000 steps at lr=2⁻⁶. Confirm
this rather than trusting it: every run logs `cumulative_pruned` and
`cumulative_regrown`, and those should be in the same ballpark as SET's. If
the whole grid is flat, extend upward before concluding anything.

**Published SET only.** `prune_metric: magnitude`. The codebase also has
`utility`, which beat magnitude 9.26 vs 15.41 at 8 tasks in the old repo — but
that is a different algorithm, not SET, and is out of scope here.

Analysis: `analysis/04_connectivity_methods.ipynb` (not written yet).
