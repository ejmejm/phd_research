# Debug Task Separation

Why does pruning+generation achieve task specialization on a toy problem
but fail on parallel MNIST? These experiments systematically identify the
causes and fix them.

## Background

A 1-hidden-layer network with 2 independent tasks learns to dedicate hidden
units to one task each via utility-based pruning and generation (see
`outgoing_pruning_gen.ipynb`). Running the same idea at MNIST scale via
`column_guided_search.py` fails: `input_entropy` stays ~0.35.

## Scaling Progression (03)

Starting from the working toy setup, we incrementally make the problem more
MNIST-like. All use contribution utility, SGD, stationary data, 3x turnover,
0.5% batch, EMA utility with median reset, 20 seeds. Partial connectivity
steps use spp=8000 (see "What Broke" below for why).

| Step | Change | Purity | Entropy | Loss | Steps |
|:--:|:--|:--:|:--:|:--:|:--:|
| 1 | Baseline toy (2in/2out, MSE) | 0.975 +/- 0.014 | 0.053 | 0.254 | 1M |
| 2 | + Noisy targets | 0.953 +/- 0.014 | 0.099 | 0.415 | 1M |
| 3 | + Classification (CE, 2 class) | 0.867 +/- 0.024 | 0.283 | 0.019 | 1M |
| 4 | + 10 classes per task | 0.990 +/- 0.014 | 0.024 | 0.436 | 1M |
| 5 | + 100 inputs/task, full conn | 0.964 +/- 0.012 | 0.103 | 1.282 | 1M |
| 6 | + 100 inputs/task, partial 64 | 0.992 +/- 0.000 | 0.016 | 1.365 | 4.8M |
| 7 | + 784 inputs/task, partial 64 | 0.986 +/- 0.004 | 0.035 | 2.105 | 4.8M |
| 8 | + Nonlinear teacher (MLP) | 0.979 +/- 0.003 | 0.063 | 1.761 | 4.8M |

All steps pass (purity > 0.86). Notes on steps that needed fixes:

- **Step 3** (CE, 2 classes): moderate drop to 0.867. Only 2 output neurons
  per task gives a weak gradient signal. Recovers at step 4 with 10 classes.
- **Step 5** (100 inputs): originally failed at 0.506 — not enough total
  pruning. Fixed with batch pruning for adequate turnover (see below).
- **Step 6** (partial 64 conns): originally 0.863 even with 3x turnover.
  Fixed by increasing training between prune events to spp=8000 (see below).
- **Steps 7-8**: worked with step 6's settings, no additional tuning needed.

## What Broke Along the Way

### Insufficient turnover at scale (04, 05)

Step 5 (100 inputs/task, full connectivity) failed at 0.506 purity when
using the toy's pruning settings (1 connection pruned per 5000-step cycle,
200 cycles). The toy achieves 10x turnover of its 20-connection budget;
the scaled problem with budget=5500 got 0.04x turnover from the same 200
prune events.

The utility signal was fine — even 25M training steps didn't help (04-A).
The fix was simply more total pruning. **Connection turnover is the key
variable** (04):

| Turnover | Purity | Example |
|:--:|:--:|:--|
| 0.04x | 0.51 | 200 prunes of 1 |
| 0.4x | 0.64 | 200 prunes of 10 |
| 1.8x | 0.98 | 200 prunes of 50 |
| 10.0x | 0.98 | 1000 prunes of 55 |

Batch size itself doesn't matter — at fixed 1.8x turnover, pruning 1/event
over 10000 events (0.963) works about as well as 50/event over 200 events
(0.981). The only failure is pruning >3% of the budget at once (05).

### Insufficient training with partial connectivity (06)

Step 6 (100 inputs, partial 64 conns) dropped to 0.863 even with 3x
turnover. Unlike full connectivity where more training between prunes
doesn't help, **partial connectivity needs more spp** — each unit sees
only a fraction of inputs, so the utility signal is noisier and takes
longer to stabilize.

| Experiment | Purity | Key |
|:--|:--:|:--|
| Baseline (spp=1666) | 0.838 | |
| 10x turnover (spp=1666) | 0.869 | More pruning alone barely helps |
| **5x spp (spp=8333)** | **0.993** | **Same turnover, more training** |
| Smaller budget (1000) | 0.961 | Fewer units = stronger per-unit signal |
| budget=10000 | 0.717 | Too many units, signal too diluted |

Smaller budgets also help: fewer units means the gradient concentrates on
fewer connections. Budget=400 (4 units) reached 0.954 vs budget=5000 (59
units) at 0.838 with identical turnover and spp.

## Real MNIST (07)

With proper pruning (3x turnover, 0.5% batch, spp=8000, EMA + median
reset), we test the original algorithmic variants on real MNIST (2 tasks,
784 inputs/task, 10 classes/task, partial 64 conns, budget=5000, 20 seeds).

| Experiment | Purity | Entropy | Loss |
|:--|:--:|:--:|:--:|
| Baseline (SGD) | 0.655 +/- 0.012 | 0.878 | 0.302 |
| Propagated utility | 0.670 +/- 0.018 | 0.837 | 0.314 |
| Single output gen | 0.816 +/- 0.027 | 0.534 | 0.643 |
| **Adam optimizer** | **0.979 +/- 0.006** | **0.052** | **0.535** |
| Non-stationary (permute=2000) | 0.600 +/- 0.004 | 0.945 | 2.312 |
| **Prop + single + Adam + nonstat** | **0.977 +/- 0.006** | **0.057** | 3.211 |

On synthetic data (scripts 01, 03), all algorithmic variants performed
equivalently. On real MNIST, **Adam is critical** — it jumps from 0.655
(SGD) to 0.979. Adam's adaptive learning rates help each connection develop
meaningful weight magnitudes faster, producing a clearer utility signal
within the 8000-step window between prune events. SGD with the same spp
doesn't develop weights fast enough for the utility to reliably distinguish
task-relevant from cross-task connections.

Non-stationarity hurts with SGD (0.600) but Adam compensates (0.977 even
with permute=2000). Propagated utility and single-output generation make
little difference — Adam is the dominant factor.

## Open Questions

- Do Adam's separated units have very few input connections, or do they
  maintain full 64-connection inputs that are task-pure? If units are
  collapsing to a handful of connections, that's a different kind of
  "separation" (sparsity) rather than task-driven sculpting.
- Can SGD match Adam with even more spp, or is the per-step learning
  rate dynamics fundamentally different?
- How do these findings translate back to the actual DynamicNetwork
  implementation in `column_guided_search.py`?

## Scripts

| Script | Purpose |
|:--|:--|
| 01_toy_algorithmic.py | Algorithmic variants on toy problem (all pass) |
| 02_mnist_scale.py | Same variants at MNIST scale with bad pruning (all fail) |
| 03_scaling_progression.py | Toy → MNIST-like progression with proper pruning |
| 04_pruning_dynamics.py | Turnover, batch size, training time sweeps |
| 05_batch_size_ablation.py | Fixed turnover, varying batch size (batch doesn't matter) |
| 06_partial_conn_tuning.py | Partial connectivity tuning (spp is key) |
| 06b_small_budget.py | Very small budget sweep |
| 07_mnist_with_fixes.py | Real MNIST with all pruning fixes applied |
