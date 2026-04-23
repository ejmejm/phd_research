# Local Pruning Progression

A progression of experiments that rethinks connection-level pruning from the
neuron's perspective, starting from the simplest possible setup (linear,
global pruning) and increasing complexity step by step. Follow-on steps will
add hidden units, per-unit local budgets, and dynamic budget allocation.

Builds on findings from `../debug_task_separation/`.

## Metrics

All metrics are reported as mean ± 95% CI over 20 seeds unless otherwise
noted. The architecture is a linear model: `logits = (W * M) @ x`,
reshaped to (N_TASKS=2, 10) for per-task softmax CE. `W, M : (20, 1568)`.
Inputs are two concatenated MNIST digits (784 + 784 = 1568). Outputs are
20 neurons: 10 per task (one per digit class).

### Loss

- **final_loss** — mean training cross-entropy loss over the final **40k
  steps** of the run. Measures the network's rate of adaptation under
  non-stationarity (label permutations every 4000 steps), not peak
  accuracy. Lower is better.

### Connectivity / structure metrics

These measure what the final connectivity mask `M` looks like — which
(output, input) slots are active and whether they respect task boundaries.
Each output neuron `k` serves a specific task `t = k // 10`. Each input
`j` belongs to a specific task `t = j // 784`. A connection `(k, j)` is
**task-aligned** when `k // 10 == j // 784`.

- **alignment** (= precision) — fraction of active connections that are
  task-aligned: `TP / (TP + FP)` where TP = active and aligned, FP =
  active and cross-task. Chance level = 0.5, oracle = 1.0. Introduced in
  step 2. Can be gamed by pruning everything except a few aligned
  connections.
- **separation_f1** — F1 score treating connection activity as a binary
  classifier of "is this slot task-aligned?":
  - TP = active & aligned
  - FP = active & cross-task
  - FN = inactive & aligned (missed same-task coverage)
  - F1 = 2·P·R / (P+R) where P = alignment, R = TP / (TP+FN)

  Penalizes both cross-task noise (low P) and missing same-task coverage
  (low R). Can't be gamed by sparse aligned-only masks. Introduced in
  step 4. Reference points: oracle intask at budget=1500 → F1=0.175;
  fully connected → F1=0.667. Not meaningful at fixed low budgets
  (e.g. 1500) because recall is capped at budget / n_possible_aligned.
- **purity** — per-unit metric. For output neuron k, count active
  connections from each task block: `t0 = M[k, 0:784].sum()`,
  `t1 = M[k, 784:1568].sum()`. Purity of that unit =
  `max(t0, t1) / (t0 + t1)`. Average across units (skip units with no
  active connections). Chance ≈ 0.5, oracle = 1.0.
- **entropy** — per-unit metric. Normalized binary entropy of the same
  `(t0, t1)` split: `H = −Σ p·log₂(p) / log₂(2)` where `p = (t0, t1)
  / (t0+t1)`. Averaged across units. Perfect separation → 0.0, fully
  mixed → 1.0.

### Budget / convergence metrics (steps 4+)

These apply to experiments that start fully connected and prune down to
a natural size.

- **final_budget** — number of active connections at convergence (when 3
  consecutive prune events prune 0 connections). Per-seed, indexed by
  converge_cycle.
- **converge_step** — training step at which 3 consecutive zero-prune
  events is first reached. Sentinel value = total steps if convergence
  never happens.

### Trajectory metrics (logged to MLflow)

- **loss_window_5k** — mean loss in each 5k-step window, for convergence
  inspection.
- **n_pruned_window** / **n_generated_window** — total connections
  pruned / generated per 5k-step window (mean across seeds).
- **n_active** — active connection count at end of each 5k-step window
  (mean across seeds).
- **demand_unit_{00..19}** (step 7) — per-output-neuron demand `d_i` at
  end of each 5k-step window (mean across seeds). Shows which neurons
  the generation policy is favoring over time.

## Invariants (constant across every step)

These hold for every experiment in this directory.

- **Task**: non-stationary parallel multi-MNIST, 2 sub-tasks. Inputs are
  concatenated MNIST digits (1568 = 2 × 784). Labels are two class indices
  in [0, 10). Every **4000 steps** one randomly-chosen task's label mapping
  is replaced with a fresh random permutation of [0..9]. Non-stationarity
  is there to overcome the blocking problem. Steps 1-5 use raw MNIST
  (pixels in [0, 1]); **steps 6-7 use per-pixel standardized MNIST**
  (mean 0, std ≈ 1) so the statistical-threshold formula can assume
  σ_x = 1.
- **Batch size**: 1 (online).
- **Seeds**: 20 seeds per reported configuration, base seed 42.
- **Run length**: 225k training steps for all steps.
- **Evaluation**: running average of training cross-entropy loss over the
  final **40k steps** of the run. This measures rate of adaptation under
  non-stationarity, not peak accuracy.
- **Reported metrics** (for every run, mean ± 95% CI over seeds):
  `final_loss`, `alignment`, `purity`, `entropy`, and step-specific
  metrics as applicable. See **Metrics** section above for definitions.
- **Baselines** (steps 1-3 only; logged as their own MLflow runs):
  1. **Random fixed connectivity** — 1500 connections sampled uniformly
     from the full (20 × 1568) slot pool, frozen for the whole run. Train
     weights with SGD; no pruning/generation.
  2. **Within-task fixed connectivity** — 750 connections per task sampled
     uniformly from within that task's (10 × 784) block, frozen for the
     run.

  Steps 4+ change the paradigm (fully-connected init, threshold pruning,
  normalized inputs) and do not re-run these baselines.
- **Learning-rate sweep**: every step does its own LR sweep since gradient
  scale changes with init, budget, and input normalization. Steps 1-3 use
  a factor-of-4 geometric grid with 5 seeds per LR, then re-run the
  winner with 20 seeds. Steps 4+ use powers-of-2 grids with 20 seeds
  directly (via mlflow-sweeper).
- **MLflow**: project name `local_pruning_progression`. Steps 1-3 name
  runs as `step{N}_{variant}` (e.g. `step1_dynamic`). Steps 4+ use
  mlflow-sweeper with sweep names like `step{N}_{sweep_name}` and
  per-trial runs tagged with parameter values. Logs aggregated (mean + CI)
  metrics across seeds — no per-seed child runs. Also logs a
  moving-window loss trajectory (`loss_window_5k` every 5k steps) for
  convergence inspection.

## Step 1 — Linear, global pruning

**Goal**: minimal-setup baseline for the progression. No hidden units;
pruning decisions and the budget are both global.

- Architecture: linear. `logits = (W * M) @ x`, reshape to `(N_TASKS=2, 10)`
for per-task softmax CE. `W, M : (20, 1568)`.
- Budget: 1500 active connections (≈ 4.8% density, ≈ 1/10 of within-task
dense).
- Init: 1500 randomly placed connections.
- Every step: SGD `W ← W − lr · ∇loss · M`.
- Utility per connection: EMA, `γ = 0.998`, of `|x| · |W| · M`.
- Prune event every **50 steps**: remove the 1 connection with the lowest
EMA utility (break ties with tiny random noise).
- Generation at each prune event: add 1 new connection in a
uniformly-random currently-empty (output, input) slot. New connection
weight = 0. **New connection utility = mean of currently-active
utilities**.
- Run length: **225,000** steps ⇒ 4500 prune events ⇒ 3× turnover of the
1500-connection budget.

### Results

LR swept on a factor-of-4 grid from 1.56e-4 to 2.56; all three variants
had the same best LR (0.16), a clear U-shape on either side. Finals at
the best LR, 20 seeds, 225k steps (3× turnover of 1500 connections):


| Variant                                | MLflow run           | LR   | Final-40k loss    | Purity        | Entropy       |
| -------------------------------------- | -------------------- | ---- | ----------------- | ------------- | ------------- |
| Dynamic (prune+gen, 1 conn / 50 steps) | `step1_dynamic`      | 0.16 | **1.040 ± 0.007** | 0.578 ± 0.006 | 0.975 ± 0.003 |
| Fixed random                           | `step1_fixed_random` | 0.16 | 1.404 ± 0.016     | 0.542 ± 0.003 | 0.992 ± 0.001 |
| Fixed within-task                      | `step1_fixed_intask` | 0.16 | **1.016 ± 0.009** | 1.000 ± 0.000 | 0.000 ± 0.000 |


Full LR sweep curves (5-seed means):


| LR         | Dynamic loss | Random loss | Intask loss | Dynamic purity |
| ---------- | ------------ | ----------- | ----------- | -------------- |
| 1.56e-4    | 2.269        | 2.286       | 2.269       | 0.678          |
| 6.25e-4    | 2.202        | 2.253       | 2.202       | 0.689          |
| 2.5e-3     | 1.957        | 2.140       | 2.008       | 0.678          |
| 1e-2       | 1.519        | 1.895       | 1.646       | 0.652          |
| 4e-2       | 1.127        | 1.584       | 1.249       | 0.610          |
| **1.6e-1** | **1.042**    | **1.395**   | **1.011**   | 0.579          |
| 6.4e-1     | 2.391        | 1.682       | 1.149       | 0.559          |
| 2.56       | 10.3         | 5.09        | 3.44        | 0.558          |


### Interpretation

1. **The dynamic method matches the within-task oracle on loss** (1.040
  vs 1.016, gap within CI noise), and clearly beats the random-fixed
   baseline (1.404). So connection-level pruning *is* finding a
   genuinely useful subset of 1500 connections — better than random.
2. **But it does not achieve task separation at the connectivity level.**
  Dynamic purity is 0.578, barely above random's 0.542. The within-task
   oracle has purity 1.0 and similar loss. So "what matters for loss"
   and "whether connections are task-aligned" are decoupled here.
3. **The utility signal is selecting informative pixels, not task-aligned
  ones.** Contribution utility `|x| · |W| · M` is high for pixels that
   are active and have non-trivial weights. MNIST's center pixels meet
   this bar regardless of which task the output neuron serves; edge
   pixels don't. Random-fixed wastes connections on edges; within-task
   fixed puts them all on center-ish pixels of the *right* task; dynamic
   puts them on center-ish pixels of *either* task — which is nearly as
   good because cross-task connections develop small weights and
   contribute little gradient noise at steady state.
4. **Dynamic purity actually drops as LR rises** (0.678 at lr=1.56e-4 →
  0.558 at lr=2.56). At low LR, weights barely move and utility ≈
   `|x| * |W_init| = |x| * 0`, so the prune order is determined by tiny
   initial noise — which happens to average toward slightly higher
   task-alignment. At high LR, weights change rapidly and utility
   follows; but the signal reflects *what's informative*, not *what's
   task-local*. The sweet spot for loss (0.16) is well past the
   (spurious) sweet spot for purity.
5. **This cleanly replicates the `debug_task_separation/` finding** —
  contribution-utility pruning on MNIST doesn't drive task separation
   without extra pressure (the "Adam compensates" result was already
   really "Adam + sparsity collapse"). The linear setup makes this
   visible without being confounded by hidden-unit dynamics. The
   follow-on steps in this progression will introduce mechanisms that
   should actually drive task separation.

## Step 2 — Budget sweep

**Hypothesis**: step 1's dynamic method failed to separate because at
budget=1500 it had enough slack to keep both the "informative pixel"
signal and the "task-alignment" signal in its selection. If the budget
is small, dynamic is forced to pick task-aligned connections because
cross-task connections bring no signal and the opportunity cost of
wasting a slot rises. Smaller budget ⇒ higher task alignment.

Same setup as step 1: linear, non-stationary multi-MNIST, 20 seeds,
225k steps, lr=0.16, spp=50 (1 connection pruned every 50 steps). Only
the budget varies. Because pruning is at a fixed absolute rate,
smaller budgets see much higher turnover (4500 prune events / budget).

### Metric note

At very small budgets (≤50), most output neurons have 0–1 connections,
which makes the per-unit purity metric degenerate (a unit with 1
connection always has purity = 1.0). We therefore report **task
alignment** — the fraction of all active connections whose input
comes from the same task as their output — as the primary separation
metric alongside loss. Alignment's chance level is 0.5 and its oracle
value is 1.0, regardless of budget.

### Results (20 seeds, 95% CI)


| Budget   | Turnover | Variant      | Loss              | Alignment         | Purity |
| -------- | -------- | ------------ | ----------------- | ----------------- | ------ |
| **1500** | 3×       | dynamic      | **1.040 ± 0.007** | 0.572 ± 0.006     | 0.578  |
|          |          | fixed random | 1.404 ± 0.016     | 0.500 ± 0.004     | 0.542  |
|          |          | fixed intask | **1.016 ± 0.009** | 1.000             | 1.000  |
| **500**  | 9×       | dynamic      | **1.206 ± 0.008** | 0.767 ± 0.006     | 0.770  |
|          |          | fixed random | 1.887 ± 0.019     | 0.498 ± 0.009     | 0.581  |
|          |          | fixed intask | 1.580 ± 0.018     | 1.000             | 1.000  |
| **150**  | 30×      | dynamic      | **1.725 ± 0.014** | **0.832 ± 0.011** | 0.835  |
|          |          | fixed random | 2.168 ± 0.011     | 0.487 ± 0.014     | 0.662  |
|          |          | fixed intask | 2.060 ± 0.016     | 1.000             | 1.000  |
| **50**   | 90×      | dynamic      | **2.086 ± 0.008** | 0.792 ± 0.023     | 0.872  |
|          |          | fixed random | 2.264 ± 0.006     | 0.490 ± 0.023     | 0.777  |
|          |          | fixed intask | 2.228 ± 0.006     | 1.000             | 1.000  |
| **20**   | 225×     | dynamic      | 2.217 ± 0.006     | 0.745 ± 0.037     | 0.923  |
|          |          | fixed random | 2.289 ± 0.003     | 0.477 ± 0.039     | 0.897  |
|          |          | fixed intask | 2.275 ± 0.003     | 1.000             | 1.000  |


All runs logged to MLflow as `step2_{variant}_budget{B}` under project
`local_pruning_progression`.

### Interpretation

1. **Budget pressure drives task alignment — exactly as hypothesized.**
  Dynamic alignment rises from **0.57** at budget=1500 to **0.83** at
   budget=150, peaking in the middle of the range. The random baseline
   stays at ~0.50 (chance) across all budgets. So the observed shift is
   a real behavior of the algorithm, not an artefact of the metric.
2. **At budgets 500 and 150, dynamic beats the within-task oracle on loss.**
  This is the most interesting finding: 1.206 (dynamic) vs 1.580
   (intask) at budget=500; 1.725 vs 2.060 at budget=150. Intask-random
   samples connections from the correct task but at random pixel
   positions, so it wastes slots on edge (inactive) pixels. Dynamic
   trades a small loss on task alignment (~0.17 of its slots go
   cross-task) for a much bigger win on pixel informativeness. The
   two signals stack.
3. **At budget=20, all three converge.** Loss ≈ 2.23, essentially
  the unconditional log(10) = 2.30 floor. The model is too small to
   fit the task regardless of which connections it has, so any signal
   from either axis is drowned out.
4. **Alignment is non-monotonic in budget** — peaks at budget=150,
  dips back at budget=20. At extremely small budgets, weight magnitudes
   never develop meaningfully (too few connections → gradients are
   tiny per-step), so utility is noisier and drives less separation.
5. **The "sweet spot" for combined separation + capacity is around
  budget 150–500.** This is where the alignment signal is maximal
   AND the network still has enough capacity to produce useful loss.
   For follow-on steps (hidden layers, per-unit budgets), budget=150
   or 500 is a more discriminating operating point than 1500.

## Step 3 — Signed utility

**Hypothesis**: the signed LOO utility `U = |e_k + c[k,i]| − |e_k|`,
where `e_k = onehot_k − softmax_k` and `c = W[k,i] · x[i]`, should
give cross-task connections with active inputs negative utility (c is
uncorrelated with e, so |e+c|−|e| fluctuates sign-wise and EMAs toward
~0 or negative), while aligned connections accumulate positive U.

Same setup as step 2 (same budget sweep, same lr=0.16, same 225k steps),
just swapping contribution utility for signed utility in the dynamic
method. Fixed baselines are unchanged (they don't use utility).

### Results (20 seeds, 95% CI)

Side-by-side with step 2's contribution utility:


| Budget | Utility      | Loss          | Alignment     |
| ------ | ------------ | ------------- | ------------- |
| 1500   | contribution | 1.040 ± 0.007 | 0.572 ± 0.006 |
| 1500   | signed       | 1.038 ± 0.008 | 0.575 ± 0.006 |
| 500    | contribution | 1.206 ± 0.008 | 0.767 ± 0.006 |
| 500    | signed       | 1.217 ± 0.013 | 0.770 ± 0.006 |
| 150    | contribution | 1.725 ± 0.014 | 0.832 ± 0.011 |
| 150    | signed       | 1.764 ± 0.012 | 0.814 ± 0.009 |
| 50     | contribution | 2.086 ± 0.008 | 0.792 ± 0.023 |
| 50     | signed       | 2.102 ± 0.010 | 0.774 ± 0.022 |
| 20     | contribution | 2.217 ± 0.006 | 0.745 ± 0.037 |
| 20     | signed       | 2.222 ± 0.006 | 0.733 ± 0.038 |


All runs logged as `step3_dynamic_signed_budget{B}`.

### Interpretation

**Signed utility does not improve over contribution utility.** The two
are essentially identical across all budgets — differences are within CI
noise and if anything signed is slightly worse (lower alignment at
budget=150: 0.814 vs 0.832).

## Step 4 — Threshold pruning from fully connected

**Hypothesis**: the prior steps are unstable because they always prune the
*globally lowest-utility* connection, no matter how useful it actually is —
the network is in constant churn. Step 4 swaps that for a **fixed threshold
at zero**: start fully connected, and at each prune event remove every
connection with signed EMA utility ≤ 0. No generation. The network stops
pruning once no harmful connections remain (3 consecutive prune events
with zero pruned), and then trains without any further topology changes.
The final connectivity is whatever the network naturally settles on.

Setup: linear, non-stationary multi-MNIST, 20 seeds, 4500 max cycles
(225k steps), SPP=50, signed utility, EMA γ=0.998. Init: M = ones
(all 31,360 connections active). After convergence, training continues
for the eval window. LR swept separately since the fully-connected
gradient scale is very different from the budget=1500 regime.

### Separation metric note

Alignment (precision = TP / (TP+FP)) can be gamed by pruning everything
except a handful of correctly-aligned connections. Step 4 therefore adds
**separation F1**: treat each (output, input) pair as a classification
problem where the label is "input's task == output's task" and the
prediction is "connection active". Then:

- TP = active connection & input in right task
- FP = active connection & input in wrong task
- FN = inactive connection & input in right task (missed coverage)
- Precision = TP/(TP+FP) = current alignment metric
- Recall = TP/(TP+FN) = fraction of same-task inputs that are connected
- F1 = harmonic mean of the two — penalizes both cross-task noise *and*
missing same-task coverage.

Reference points (20 outputs × 1568 inputs, 15,680 possible aligned):
oracle intask at budget=1500 → F1=0.175 (perfect precision, limited
coverage); fully connected → F1=0.667 (perfect recall, half precision);
"gamed" 1 aligned connection → F1=0.0001.

### LR sweep (20 seeds, 95% CI)

Powers of 2 centered on 2^-5 = 0.03125.


| LR                | Loss              | Alignment (P) | Sep-F1            | Purity | Budget      | Converge cycle |
| ----------------- | ----------------- | ------------- | ----------------- | ------ | ----------- | -------------- |
| 2^-7 (0.0078)     | 1.987 ± 0.055     | 0.828 ± 0.011 | 0.153 ± 0.023     | 0.822  | 1615 ± 265  | 791 ± 50       |
| 2^-6 (0.0156)     | 1.668 ± 0.060     | 0.857 ± 0.009 | 0.180 ± 0.024     | 0.837  | 1874 ± 284  | 892 ± 81       |
| **2^-5 (0.0313)** | **0.985 ± 0.011** | 0.713 ± 0.008 | **0.381 ± 0.008** | 0.716  | 5757 ± 241  | 500 ± 69       |
| 2^-4 (0.0625)     | 1.207 ± 0.011     | 0.503 ± 0.000 | 0.548 ± 0.002     | 0.508  | 18765 ± 152 | 49 ± 11        |
| 2^-3 (0.125)      | 2.398 ± 0.021     | 0.500 ± 0.000 | 0.550 ± 0.002     | 0.507  | 19153 ± 161 | 18 ± 2         |


All runs logged as `step4_threshold_lr_sweep_lr={value}` under project
`local_pruning_progression`.

### Interpretation

1. **Best LR drops ~5×** from step 1. Previous steps peaked at lr=0.16;
  step 4 peaks at **lr=2^-5 ≈ 0.031** and lr=2^-3 = 0.125 already collapses
   (loss 2.40, essentially uniform). The fully-connected init has ~21× more
   active connections than budget=1500, so the summed gradient is much larger
   and a smaller LR is needed to avoid weight blow-up.
2. **Step 4 roughly matches step 1's dynamic method on loss at its best LR:**
  0.985 vs 1.040. The network keeps ~5,800 connections — about 4× the
   budget=1500 setting — so the capacity-vs-sparsity trade-off tilts in
   step 4's favor. Not apples-to-apples with steps 1–3, but it demonstrates
   that the threshold mechanism finds a useful operating point without
   needing a hand-picked budget.
3. **Final budget is strongly LR-dependent.** Low LR (2^-7–2^-6) prunes
  aggressively to ~~1,600–1,900 connections because weights barely move and
   most connections accumulate near-zero (hence non-positive) utility. High
   LR (2^-4–2^-3) barely prunes at all (~~19,000 kept) because large weight
   magnitudes make almost every contribution look useful. Medium LR (2^-5)
   lands in between: enough signal to distinguish useful from useless,
   enough pressure to prune the latter.
4. **Convergence happens well within the run budget** for all viable LRs —
  cycle 500 at lr=2^-5 (25k steps), and never later than cycle ~900. The
   eval window (last 800 cycles = 40k steps) is always fully post-convergence
   at the best LR, so the reported losses reflect the **stable pruned
   network**, not a still-changing one.
5. **Alignment is misleading; separation-F1 tells the real story.** Low-LR
  configs look great on alignment (0.83–0.86) but their F1 is terrible
   (0.15–0.18) — they prune so aggressively that they only cover a tiny
   fraction of useful same-task inputs, so the "high alignment" is mostly
   a small-budget artefact. At the best-loss lr=2^-5, alignment is lower
   (0.713) but F1 is the highest among trained configurations (0.381) —
   the network keeps ~2× more of the possible aligned connections than
   the low-LR regime while still having ~71% of its connections aligned.
   The high-LR configs (2^-4, 2^-3) have higher F1 (0.548, 0.550) than lr=2^-5
   only because they're essentially fully connected, which is the F1
   ceiling for a non-separating network (P=0.5, R=1.0).
6. **The stability goal is met.** At lr=2^-5, the topology freezes at cycle
  500 (25k steps); the remaining 200k steps train under a fixed network.
   No churn. This is a qualitatively different operating regime from the
   constantly-rewiring dynamic methods in steps 1–3.

## Step 5 — SPP sweep (steps between prune events)

**Hypothesis**: step 4 prunes every 50 training steps. That's not much
warmup before the first pruning decision — many genuinely-useful
connections may get pruned simply because their EMA utility hadn't had
time to develop a reliable signal. Longer SPP ⇒ more warmup per decision
⇒ fewer unlucky prunes ⇒ better final network.

Same setup as step 4 (fully connected init, signed utility, threshold ≤ 0,
no generation, 3-consecutive-zero stop), at the best LR from step 4
(2^-5 = 0.03125). **Total training steps held constant at 225k** across
the sweep — so larger SPP means fewer prune events, not more training.

Each cycle runs `spp` training steps BEFORE pruning, so the first prune
event is at step `spp` (not step 0); the EMA always has ≥ spp steps of
warmup.

### Results (20 seeds, 95% CI)


| SPP     | n_cycles | Loss              | Alignment (P) | Sep-F1        | Budget      | Converge step |
| ------- | -------- | ----------------- | ------------- | ------------- | ----------- | ------------- |
| 50      | 4500     | 0.985 ± 0.011     | 0.713 ± 0.008 | 0.381 ± 0.008 | 5757 ± 241  | 25012         |
| 100     | 2250     | 0.958 ± 0.009     | 0.725 ± 0.007 | 0.367 ± 0.008 | 5338 ± 204  | 57580         |
| 200     | 1125     | 0.835 ± 0.012     | 0.647 ± 0.004 | 0.435 ± 0.007 | 7943 ± 212  | 80260         |
| **400** | 562      | **0.784 ± 0.007** | 0.571 ± 0.002 | 0.507 ± 0.004 | 12516 ± 135 | 140540        |
| 800     | 281      | 0.823 ± 0.008     | 0.522 ± 0.001 | 0.527 ± 0.001 | 16005 ± 64  | 223480        |


All runs logged as `step5_spp_sweep_spp={value}` under project
`local_pruning_progression`.

### Interpretation

1. **Hypothesis confirmed: longer SPP gives lower loss**, up to SPP=400.
  Loss drops monotonically from 0.985 (SPP=50) to 0.784 (SPP=400) — a
   ~20% improvement — before rising slightly at SPP=800 (0.823).
2. **F1 rises monotonically with SPP** (0.381 → 0.527). Longer warmup
  means more genuinely-useful connections survive each prune event,
   which directly improves recall (coverage of same-task inputs). The
   F1 gain comes from less aggressive false-negative pruning, not from
   better precision — if anything alignment (= precision) drops with
   SPP (0.713 → 0.522) because the larger budgets carry more cross-task
   noise.
3. **Final budget grows with SPP** (5757 → 16005). The threshold mechanism
  only prunes what it has clear evidence is harmful; with more warmup
   and fewer prune events, fewer borderline connections get flagged.
   The "natural size" the network settles on is essentially set by SPP
   at this LR.
4. **At SPP=800, convergence happens very late (step 223,480 of 225,000).**
  Only 1,520 steps of post-convergence training — the eval window (last
   40k steps) almost entirely overlaps with the pruning phase. SPP=800
   essentially never finished pruning within the run budget, so its
   loss of 0.823 is measured during active pruning, not on a stable
   network. The fact that it is still competitive with SPP=400 despite
   this handicap suggests that the "more warmup ⇒ better" trend likely
   continues past SPP=400 if given enough total steps. SPP=400 wins here
   partly because it's the largest SPP where the network actually
   finishes pruning comfortably before the eval window, not necessarily
   because it's the true optimum.
5. **This is a genuine stability-vs-accuracy trade-off.** Short-SPP
  regimes prune aggressively at every opportunity and churn out a
   small, tightly-aligned network (better alignment, worse loss and F1).
   Long-SPP regimes are more conservative, keeping connections that
   *might* be useful, and end up with a larger, less-aligned network
   that generalizes better under non-stationarity.
6. **Takeaway: this works, but requires manual tuning.** Getting a good
  network out of step 5 meant discovering by sweep that SPP=400 was the
   right cadence for this LR and this total-step budget. The best SPP
   isn't knowable a priori — it depends on LR, utility noise scale, and
   total training steps. Manually recalibrating SPP for every new
   configuration is brittle. This motivates a method that decides *when
   the utility estimate is reliable enough to act on* from the statistics
   of the estimate itself, rather than from a hand-tuned fixed cadence.

## Step 6 — Statistical-confidence threshold pruning

**Hypothesis**: step 5 showed that longer SPP helps because short SPP
doesn't give the EMA time to develop a reliable signal — connections
get unluckily pruned for noisy-negative utility before they've had a
fair chance to show usefulness. But the right SPP is not knowable a
priori. Step 6 replaces the fixed τ ≤ 0 rule with a **per-weight
statistical confidence threshold** that widens when the estimate is
uncertain and tightens as the EMA accumulates samples. The prune
decision becomes "only remove the weight if we're 1 − α confident its
utility is genuinely negative, not just noisy."

### Method

For each weight w, track the EMA utility Û and bias-correct:

```
Û_corr = Û / (1 − β^t)
```

where t is the age of the weight (= step count, since weights are
never regenerated here). Prune iff Û_corr < τ_w, with

```
τ_w(t) = −z_α · |w| · σ_x · sqrt(K · (1 + β^t) / (1 − β^t))
```

and K = (1 − 2/π)·(1 − β)/(1 + β) precomputed once at load. The
`|w|·σ_x` factor scales the threshold with the noise level of the
utility samples — larger contributions have noisier estimates and need
more evidence. z_α is specified by the user as a one-sided confidence
level (0.9 → 1.28, 0.95 → 1.64, 0.99 → 2.33).

**Input normalization**: to get σ_x = 1 into the formula without
per-weight bookkeeping, MNIST is standardized upfront — each pixel is
made zero-mean and unit-variance over the training set (std floored at
1e-3 to avoid amplifying near-constant pixels). Everything else (signed
utility, fully connected init, β=0.998, 225k steps, convergence on 3
consecutive zero-prune events, 20 seeds) is identical to steps 4–5.

### LR sweep (ci=0.95, spp=50, 20 seeds)

Standardization inflates input magnitudes ~3× vs raw MNIST, so step 4's
best LR (2^-5) overshoots (weights blow up → final_loss ≈ 7.1). A
separate LR sweep re-tunes; clean U-shape centered on **2^-9 ≈ 0.00195**:


| LR                 | Final loss |
| ------------------ | ---------- |
| 2^-11 (0.00049)    | 1.964      |
| 2^-10 (0.00098)    | 0.966      |
| **2^-9 (0.00195)** | **0.842**  |
| 2^-8 (0.00391)     | 0.947      |
| 2^-7 (0.00781)     | 1.533      |
| 2^-6 (0.01562)     | 3.287      |
| 2^-5 (0.03125)     | 7.096      |


Logged as `step6_lr_sweep_lr={value}`.

### Main sweep: CI × SPP (20 seeds, 95% CI, lr=2^-9)


| CI      | SPP     | Loss              | Alignment     | Sep-F1        | Budget       | Converge step    |
| ------- | ------- | ----------------- | ------------- | ------------- | ------------ | ---------------- |
| 0.9     | 50      | 0.838 ± 0.007     | 0.525 ± 0.001 | 0.612 ± 0.002 | 21,915 ± 285 | 16,135 ± 2,137   |
| 0.9     | 100     | 0.833 ± 0.008     | 0.519 ± 0.001 | 0.619 ± 0.002 | 23,170 ± 223 | 37,935 ± 4,429   |
| **0.9** | **200** | **0.823 ± 0.010** | 0.514 ± 0.001 | 0.619 ± 0.002 | 23,748 ± 263 | 104,760 ± 14,798 |
| 0.9     | 400     | 0.825 ± 0.010     | 0.509 ± 0.001 | 0.621 ± 0.002 | 24,618 ± 227 | 210,180 ± 14,610 |
| 0.95    | 50      | 0.842 ± 0.008     | 0.522 ± 0.001 | 0.632 ± 0.002 | 24,009 ± 255 | 14,105 ± 1,109   |
| 0.95    | 100     | 0.842 ± 0.008     | 0.517 ± 0.001 | 0.638 ± 0.002 | 25,223 ± 219 | 31,925 ± 5,236   |
| 0.95    | 200     | 0.833 ± 0.009     | 0.512 ± 0.001 | 0.635 ± 0.002 | 25,580 ± 270 | 103,290 ± 13,083 |
| 0.95    | 400     | 0.837 ± 0.010     | 0.507 ± 0.001 | 0.636 ± 0.002 | 26,376 ± 177 | 206,540 ± 16,482 |
| 0.99    | 50      | 0.849 ± 0.008     | 0.518 ± 0.001 | 0.648 ± 0.002 | 26,164 ± 235 | 10,928 ± 1,266   |
| 0.99    | 100     | 0.854 ± 0.008     | 0.513 ± 0.001 | 0.651 ± 0.002 | 27,221 ± 146 | 27,300 ± 3,246   |
| 0.99    | 200     | 0.851 ± 0.011     | 0.509 ± 0.001 | 0.651 ± 0.002 | 27,770 ± 245 | 69,660 ± 10,347  |
| 0.99    | 400     | 0.854 ± 0.010     | 0.504 ± 0.001 | 0.649 ± 0.002 | 28,251 ± 180 | 176,180 ± 18,075 |


Logged as `step6_statistical_threshold_ci={v}_spp={v}`.

### Interpretation

1. **Hypothesis confirmed: SPP sensitivity collapses.** Step 5's loss
  ranged from 0.985 (SPP=50) to 0.784 (SPP=400) — a ~25% swing. Step 6's
   loss at a given CI varies by at most ~1.5% across SPP 50→400:

  | Method          | SPP=50 | SPP=200 | SPP=400 | Spread |
  | --------------- | ------ | ------- | ------- | ------ |
  | Step 5 (τ ≤ 0)  | 0.985  | 0.835   | 0.784   | 0.201  |
  | Step 6, CI=0.9  | 0.838  | 0.823   | 0.825   | 0.015  |
  | Step 6, CI=0.95 | 0.842  | 0.833   | 0.837   | 0.009  |
  | Step 6, CI=0.99 | 0.849  | 0.851   | 0.854   | 0.005  |

   The confidence threshold self-calibrates: short SPP → utility is
   noisy → threshold is wide → fewer unlucky prunes. Longer SPP tightens
   the threshold but also accumulates more evidence per weight, so the
   two effects balance. No manual SPP tuning required.
2. **Lower CI → slightly better loss, but modestly.** CI=0.9 averages
  ~0.830, CI=0.95 ~0.838, CI=0.99 ~0.852. More aggressive pruning
   (demanding less evidence) yields modestly sparser networks and
   modestly lower loss. The gap is small because all three CIs settle
   on 22k-28k active connections — well above the "informative"
   operating point of steps 1-3 (budget=150-500). The threshold is
   conservative overall; it only prunes clearly-harmful weights.
3. **Step 6 does not beat step 5's best on pure loss** (best 0.823 vs
  step 5's 0.784). Step 5 at hand-tuned SPP=400 is still the single
   strongest configuration. The value of step 6 is robustness: it gets
   within 5% of step 5's best *at every SPP* without needing to sweep
   SPP, whereas step 5 at SPP=50 is ~20% worse than its own best.
4. **Alignment is essentially random (~0.51).** The statistical
  threshold has no task-separation pressure — it only asks "is this
   weight's utility convincingly negative?" — and most cross-task
   connections have ambiguous near-zero utility that passes the test.
   The network keeps 22k-28k of 31k connections, so there's little
   competitive pressure between tasks.
5. **F1 is high (0.61-0.65)** because recall stays high — most
  same-task connections survive. Best step 6 F1 (0.651 at CI=0.99,
   SPP=100) is significantly higher than step 4's best-loss F1 (0.381)
   but lower than the fully-connected ceiling (0.667). So the method is
   confidently pruning *some* cross-task connections, just not enough
   to shift the alignment metric above chance.
6. **Caveat: SPP=400 convergence pushes into the eval window.** At
  SPP=400 the network converges at ~180k-210k steps of 225k total —
   the 40k eval window partially overlaps pruning. This matches step
   5's SPP=800 caveat but is milder. SPP=50-200 has a clean
   post-convergence eval window.

## Step 7 — Demand-driven connection generation

**Hypothesis**: step 6's statistical pruning self-calibrates and is nearly
SPP-independent — but it can only *remove* harmful connections. Step 7
adds a **generation policy**: each cycle, after pruning, generate exactly
`n_pruned` new connections, allocated across output neurons in proportion
to a per-neuron **demand** estimate `d_i`. Demand for neuron `i` is an EMA
of the bias-corrected utility of recently-resolved connections to that
neuron. If a neuron's recent connections turned out useful (or harmful in
their own right), it likely has more reducible error and should get more
new slots.

### Method

For each connection track per-connection **age** `A` and **resolved-flag**
`R`. Once per connection, at the first of: (a) `U_corr` crosses `±τ_w`
(symmetric two-sided test), or (b) `A ≥ n_eff = 1/(1−β) = 500`, record
its `U_corr` into the neuron's demand EMA:

```
d_i ← β_d · d_i + (1 − β_d) · U_corr     (β_d = 0.99)
```

This captures both fast-resolved connections (clearly useful or harmful)
and slow ones (ambiguous until aged out). Pruning rule and bias
correction are unchanged from step 6, but use **per-connection age** `A`
in place of the global step counter — so freshly-generated weights get
the wide CI / generous threshold appropriate for an immature EMA.

**Allocation**: two methods swept against each other:

- *clipped_linear*: `p_i = max(d_i, 0) / Σ max(d_j, 0)` (uniform fallback if all zero)
- *softmax*: `p_i = softmax(d_i / T)` with `T = 0.01`

**Generation**: Gumbel-max top-k over empty slots, weighted by
`log(p[output])`, with a tiny uniform tie-breaker. Produces exactly
`n_pruned` new active slots; new connections start with `W=U=A=R=0`.

### Setup

Budget = **1500** (matches steps 1-3). Each cycle (every SPP=50 train
steps) prunes statistically (CI=0.9), then refills back to 1500.
Active count is constant throughout — **no convergence stopping**, runs
the full 4500 cycles. Inputs are pre-normalized (same as step 6) so
σ_x = 1 in the τ formula. 20 seeds.

### LR sweep (clipped_linear, CI=0.9, SPP=50)

Step 6's best LR (2^-9) was tuned on a fully-connected init. Sparse-init
at budget 1500 has ~20× fewer connections per output → ~20× smaller
gradient sum → a larger LR is now optimal. Re-tuned grid:


| LR                 | Final loss |
| ------------------ | ---------- |
| 2^-9 (0.00195)     | 1.834      |
| 2^-8 (0.00391)     | 1.408      |
| 2^-7 (0.00781)     | 1.298      |
| **2^-6 (0.01562)** | **1.264**  |
| 2^-5 (0.03125)     | 1.332      |
| 2^-4 (0.0625)      | 1.606      |
| 2^-3 (0.125)       | 2.355      |


Clean U-shape, best at lr=2^-6. Logged as `step7_lr_sweep_lr={value}`.

### Main sweep (allocation method, lr=2^-6, 20 seeds)


| Method           | Loss          | Alignment     | Purity        | Entropy       |
| ---------------- | ------------- | ------------- | ------------- | ------------- |
| clipped_linear   | 1.264 ± 0.012 | 0.520 ± 0.005 | 0.548 ± 0.003 | 0.989 ± 0.001 |
| softmax (T=0.01) | 1.257 ± 0.012 | 0.522 ± 0.004 | 0.547 ± 0.003 | 0.990 ± 0.001 |


Logged as `step7_generation_allocation_method={value}`.

### Interpretation

1. **Allocation method makes no difference.** clipped_linear and softmax
  give effectively identical results across every metric (loss diff
   0.007, well inside CI). The reason becomes clear from the demand
   trajectories below: per-neuron demand `d_i` converges to a tight band
   around 0.0014 with std ≈ 0.0001 across all 20 output neurons. With
   such uniform demand, both allocation rules collapse to near-uniform
   sampling — softmax(d/T) with such a narrow spread is essentially flat,
   and clipped_linear with all-positive similar values normalizes to
   ~uniform.
2. **Demand fails to differentiate output neurons.** Final demand values
  (averaged across seeds, last window):
  - task-0 outputs (units 0-9): 0.0011 – 0.0016
  - task-1 outputs (units 10-19): 0.0011 – 0.0016
  - mean = 0.0014, std = 0.0001, range = (0.0011, 0.0016)
   The task structure is symmetric (each task has 10 classes, identical
   MNIST distribution), so every output neuron faces an essentially
   identical signal-noise ratio in its resolved connections. Demand is
   working as designed — it just doesn't surface meaningful asymmetry
   here. To exploit demand, the architecture would need a source of
   genuine inter-neuron heterogeneity (e.g., per-output task identity in
   the demand signal, or hidden units with shared upstream connectivity).
3. **Pruning (and therefore generation) self-stabilizes.** Per-window
  prune count drops from ~415 (5k steps) → ~5 (115k steps) → ~2.5
   (225k steps). The network reaches a near-stable connectivity even
   without an explicit convergence rule. Generation becomes irrelevant
   in steady state because there's almost nothing to refill.
4. **Loss is worse than step 1's dynamic baseline at the same budget.**
   Step 1 dynamic at budget=1500 (raw MNIST, contribution utility,
   uniform-random generation, lr=0.16): loss=1.040, alignment=0.572.
   Step 7 (normalized MNIST, statistical pruning, demand-driven
   generation, lr=2^-6): loss=1.264, alignment=0.520. The gap is
   confounded by the dataset/normalization change and the prune-rule
   change — not a clean apples-to-apples comparison. What is clean:
   demand-driven generation in this setup is no better than uniform
   random generation, because the demand signal lacks structure.

## Step 8 — Statistical threshold pruning with BCE utility

**Hypothesis**: step 6's signed softmax-error utility is coupled across
classes within a task — increasing the contribution to one class's logit
necessarily changes the softmax of its siblings, so cross-task
connections get noisy utility signals from within-task gradient churn.
A **per-target BCE utility** treats each output logit as an independent
binary classifier; cross-task inputs feed into heads whose target is
always 0 once the net has learned, giving a cleaner and more stable
negative-utility signal. Expect higher alignment / F1 at comparable
loss to step 6.

### Method

Same pipeline as step 6 (fully-connected init, bias-corrected EMA,
statistical threshold, normalized MNIST, 3-consecutive-zero stop),
only the utility changes. For connection `(k, j)` with weight `W[k,j]`
and input `x[j]`:

```
pre_act        = (W * M) @ x                    # (OUT,)
target         = one_hot(y, NUM_CLASSES).ravel() # (OUT,), two 1's
pre_act_removed = pre_act[:, None] − x[None, :] * W   # (OUT, IN)
lp  = target · log σ(pre_act)  + (1 − target) · log σ(−pre_act)
lp' = target · log σ(pre_act_removed) + (1 − target) · log σ(−pre_act_removed)
U   = (−lp') − (−lp)
```

Sign convention matches signed utility (positive = keep, negative =
prune candidate), so the step 6 threshold formula

```
τ_w(t) = −z_α · |w| · σ_x · sqrt(K · (1 + β^t) / (1 − β^t))
```

applies, but **K changes**. Signed utility's per-sample noise has the
form `|N(0,σ²)| − const`, which has variance `(1 − 2/π)·σ²`, hence
step 6 uses `K = (1 − 2/π)·(1 − β)/(1 + β)`. BCE utility's per-sample
term is `(σ(pre_act) − target)·w·x` — no absolute value, bounded
coefficient in [−1, 1], so the leading `(1 − 2/π)` drops out and
step 8 uses `K = (1 − β)/(1 + β)`. Everything else (β=0.998,
normalized MNIST so σ_x=1, 225k steps, convergence on 3 consecutive
zero-prune events, 20 seeds) matches step 6.

### Two attempts

First attempt used **BCE utility with softmax CE training loss**. Every
LR in the grid `{2^-11 … 2^-5}` collapsed to chance (loss ≈ 2.20 vs the
log 10 ≈ 2.30 ceiling). Step 9's utility comparison traced this to
utility/loss mismatch: BCE utility measures a loss the softmax-CE-
trained network was never optimizing.

Second attempt **matches BCE utility with BCE training loss**. LR
re-tuned over `{2^-11 … 2^-3}`. Main CI × SPP sweep at lr=2^-6:

| CI   | SPP | Accuracy      | BCE loss      | Alignment      | Budget        | Converge step   |
| ---- | --- | ------------- | ------------- | -------------- | ------------- | --------------- |
| 0.90 | 50  | 0.670 ± 0.002 | 8.94 ± 0.12   | 0.474 ± 0.002  | 9,934 ± 114   | 46,388 ± 2,254  |
| 0.95 | 50  | 0.671 ± 0.002 | 9.11 ± 0.10   | 0.472 ± 0.001  | 10,233 ± 112  | 48,450 ± 2,694  |
| 0.99 | 50  | 0.671 ± 0.002 | 9.32 ± 0.14   | 0.471 ± 0.002  | 10,638 ± 129  | 54,372 ± 2,729  |
| 0.95 | 200 | 0.670 ± 0.002 | 8.75 ± 0.10   | 0.456 ± 0.001  | 9,957 ± 102   | 163,240 ± 8,485 |
| 0.95 | 400 | **0.676 ± 0.001** | 9.07 ± 0.10 | 0.457 ± 0.002 | 10,357 ± 23 | 224,800 ± 0 |

Accuracy is flat across the whole CI × SPP grid (all 12 configs within
1.1% of each other), which replicates step 6's SPP-independence. LR
sweep and main sweep logged as `step8_bce_lr_sweep_lr={v}` and
`step8_bce_statistical_threshold_ci={v}_spp={v}`.

### Why these results don't really answer the question

The BCE-loss numbers look noisy and the accuracy ceiling (~0.67) is
below step 9's signed baseline at fixed budget 1500 (0.701). That's
tempting to read as "BCE utility doesn't work as well," but the
comparison is bad.

With a 2-hot-of-20 target, the **predict-everything-as-0 strategy** is
already a strong BCE solution — most output positions want to be 0
most of the time, so a heavily-pruned, near-zero-output network can
reduce BCE loss substantially while accuracy collapses. For the same
reason, BCE loss can be improved by pruning even when the pruning is
degrading the network's actual classification ability. So BCE **loss**
is not a discriminating metric here, and accuracy understates BCE
utility's quality because BCE training wasn't optimizing accuracy
in the first place. (Separately: our BCE-loss numbers of 8–9 are
actually *worse* than the all-zero baseline of ~6.93 — the label
permutations every 4k steps keep shifting which outputs want to be 1,
so nothing settles. This makes the loss yardstick doubly unreliable.)

**The real intended use case is LTU utility** — assigning per-unit
credit based on "should this hidden unit have been 1 or 0?". In that
setting, binary targets are roughly balanced per unit (each unit is 0
or 1 based on input), the predict-all-zeros shortcut doesn't exist,
and there are no softmax-style normalization effects. BCE utility is
the natural fit for that problem. Multi-MNIST classification simply
isn't a discriminating benchmark for it, so step 8's numbers shouldn't
be treated as a verdict on the utility.

The step-8 machinery (threshold formula with BCE K, BCE loss threading,
accuracy tracking) is kept in the codebase for the LTU experiments
where it belongs. The clean utility comparison on MNIST is step 9.

## Step 9 — Utility function comparison at fixed budgets

**Goal**: do the softmax-CE LOO utility and the BCE utility drive useful
pruning *at all* under step 3's paradigm (dynamic prune+replace at a
fixed budget)? Step 8 showed BCE fails under statistical-threshold
pruning with softmax CE training, but it wasn't clear whether the
problem was the utility or the utility-loss mismatch. Step 9 pulls
that apart by running four utility/loss combinations at the same setup
otherwise, and reports accuracy (argmax correctness, averaged across
both tasks) alongside the training loss so variants with different
training losses can be compared on a common yardstick.

### Variants

| Name          | Utility           | Training loss |
| ------------- | ----------------- | ------------- |
| `signed`      | signed softmax LOO (step 3 baseline) | softmax CE |
| `softmax_ce`  | **new**: closed-form softmax-CE LOO per weight | softmax CE |
| `bce_softmax` | BCE-per-target LOO | softmax CE (mismatched) |
| `bce_bce`     | BCE-per-target LOO | BCE (matched) |

### New softmax-CE LOO utility (closed form)

Removing a single weight `W[k, j]` changes only logit `k` by
`d = −x[j]·W[k, j]`. The per-task softmax LOO NLL delta then collapses
to a scalar expression (no `(OUT, IN, NUM_CLASSES)` blowup):

```
U[k, j] = −d · 𝟙[k % NUM_CLASSES == y_task(k)]
          + log(1 + p_k · (exp(d) − 1))
```

where `p_k` is the per-task softmax probability at output `k`.
Derivation: expanding `−log_softmax(logits + d·e_k)[y] − (−log_softmax(logits)[y])`
and using the logsumexp-shift-by-one-position identity
`logsumexp(L + d·e_k) = logsumexp(L) + log(1 + softmax(L)_k · (exp(d) − 1))`.
Uses `log1p`/`expm1` for numerical stability at small `d`.
Verified element-wise against brute-force LOO.

### Setup

Dynamic prune+replace exactly as step 3: SPP=50, 225k steps, EMA γ=0.998,
20 seeds, raw (un-normalized) MNIST. Budget sweep `{50, 150, 500, 1500}`
× 4 variants = 16 configs. For every (variant, budget) the LR was chosen
from a separate 5-seed sweep over
`{1.56e-4, 6.25e-4, 2.5e-3, 1e-2, 4e-2, 0.16, 0.64, 2.56}`
(logged as `step9_lr_sweep_full`). Winner by final accuracy:

| Variant       | B=50 | B=150 | B=500 | B=1500 |
| ------------- | ---- | ----- | ----- | ------ |
| `signed`      | 0.64 | 0.64  | 0.16  | 0.16   |
| `softmax_ce`  | 0.64 | 0.64  | 0.64  | 0.64   |
| `bce_softmax` | 0.16 | 0.04  | 0.04  | 0.04   |
| `bce_bce`     | 0.64 | 0.64  | 0.64  | 0.64   |

`signed`'s best LR drops 4× between small and large budgets; the other
variants have a roughly budget-independent optimum.

### Main results (20 seeds, 95% CI)

| Variant       |    B |  lr  | **Accuracy**      | Loss               | Alignment         | Sep-F1            |
| ------------- | ---: | ---: | ----------------- | ------------------ | ----------------- | ----------------- |
| `signed`      |   50 | 0.64 | **0.242 ± 0.004** | 2.089 ± 0.007      | 0.781 ± 0.023     | 0.005 ± 0.000     |
| `softmax_ce`  |   50 | 0.64 | 0.238 ± 0.005     | 2.101 ± 0.009      | 0.748 ± 0.028     | 0.005 ± 0.000     |
| `bce_softmax` |   50 | 0.16 | 0.152 ± 0.005     | 2.197 ± 0.009      | 0.696 ± 0.020     | 0.004 ± 0.000     |
| `bce_bce`     |   50 | 0.64 | 0.141 ± 0.006     | 4.064 ± 0.033 (BCE) | 0.497 ± 0.025     | 0.003 ± 0.000     |
| `signed`      |  150 | 0.64 | **0.391 ± 0.005** | 1.788 ± 0.014      | 0.782 ± 0.014     | 0.015 ± 0.000     |
| `softmax_ce`  |  150 | 0.64 | 0.366 ± 0.007     | 1.860 ± 0.017      | 0.705 ± 0.012     | 0.013 ± 0.000     |
| `bce_softmax` |  150 | 0.04 | 0.180 ± 0.006     | 2.180 ± 0.010      | 0.623 ± 0.020     | 0.012 ± 0.000     |
| `bce_bce`     |  150 | 0.64 | 0.303 ± 0.007     | 3.204 ± 0.022 (BCE) | 0.622 ± 0.013     | 0.012 ± 0.000     |
| `signed`      |  500 | 0.16 | **0.615 ± 0.005** | 1.217 ± 0.013      | 0.770 ± 0.006     | 0.048 ± 0.000     |
| `softmax_ce`  |  500 | 0.64 | 0.477 ± 0.006     | 1.848 ± 0.021      | 0.492 ± 0.008     | 0.030 ± 0.000     |
| `bce_softmax` |  500 | 0.04 | 0.180 ± 0.007     | 2.177 ± 0.010      | 0.587 ± 0.013     | 0.036 ± 0.001     |
| `bce_bce`     |  500 | 0.64 | 0.493 ± 0.005     | 2.788 ± 0.020 (BCE) | 0.545 ± 0.007     | 0.034 ± 0.000     |
| `signed`      | 1500 | 0.16 | **0.701 ± 0.002** | 1.038 ± 0.008      | 0.575 ± 0.006     | 0.100 ± 0.001     |
| `softmax_ce`  | 1500 | 0.64 | 0.481 ± 0.005     | 1.895 ± 0.020      | 0.503 ± 0.004     | 0.088 ± 0.001     |
| `bce_softmax` | 1500 | 0.04 | 0.180 ± 0.007     | 2.176 ± 0.010      | 0.520 ± 0.006     | 0.091 ± 0.001     |
| `bce_bce`     | 1500 | 0.64 | 0.524 ± 0.005     | 3.007 ± 0.026 (BCE) | 0.474 ± 0.005     | 0.083 ± 0.001     |

Loss for `bce_bce` is BCE loss and not directly comparable to the other
three variants (all softmax CE). Accuracy is the common yardstick.

Main-run logged as `step9_main_method={m}_budget={b}`; LR sweep as
`step9_lr_sweep_full_method={m}_budget={b}_lr={lr}`.

### Interpretation

1. **Signed utility (step 3) wins at every budget**, reaching 0.701 ± 0.002
   accuracy at B=1500 — the strongest result in the progression's
   budget-sweep paradigm. It also has the only clearly budget-dependent
   optimal LR (4× drop between small and large budgets), consistent with
   the step 1 rationale: larger budget → more active connections → larger
   summed gradient → smaller LR needed to avoid weight blow-up.
2. **The closed-form softmax-CE LOO (`softmax_ce`) works, but is
   noticeably worse than signed** at every budget ≥ 150. At B=1500:
   0.481 vs 0.701 accuracy, a 22-point gap despite the utility being
   mathematically exact LOO on the training loss while signed uses the
   `|e+c| − |e|` approximation. The approximation appears to be doing
   genuine regularization: the exact softmax LOO saturates more easily
   (the `log(1 + p_k·(e^d − 1))` term is small when `p_k` is tiny, which
   is most of the time for non-target classes), so the EMA ends up with
   a weaker separation between useful and useless connections than
   signed's absolute-value-based signal.
3. **`bce_softmax` is broken** — accuracy plateaus at ~0.18 (chance ≈ 0.10)
   across every budget, training loss stuck at the softmax-CE floor.
   BCE utility and softmax-CE training cost are measuring different
   things (BCE cares about absolute logit magnitudes, softmax CE is
   shift-invariant within a task), so the utility signal drives pruning
   in a direction the training loss doesn't reward. This is the same
   pathology step 8 surfaced — now isolated cleanly.
4. **`bce_bce` works surprisingly well** — 0.524 accuracy at B=1500,
   comparable to `softmax_ce` and far above `bce_softmax`. Matching the
   utility to the training loss is what was missing in step 8. BCE
   utility *can* drive useful pruning; it just needs BCE training. Still
   noticeably below `signed` (0.701), though — the BCE training loss
   itself is probably a worse training objective for this
   non-stationary multi-task setup than softmax CE, independent of the
   pruning utility.
5. **Alignment is weakly predictive of accuracy across variants.**
   `bce_softmax` has alignment 0.52–0.70 — higher than `bce_bce` at
   B=1500 (0.47) — yet achieves one-third the accuracy. Small-budget
   alignments near 0.78 for `signed` and `softmax_ce` reflect the
   budget-pressure effect from step 2 (few slots → task-aligned
   connections are selected for because cross-task connections bring
   no signal), not a separation mechanism per se.

**Bottom line**: signed softmax LOO (step 3) remains the best utility
for this setup. Exact softmax-CE LOO works but underperforms the
approximation. BCE utility works only when paired with BCE training,
and even then underperforms softmax-based variants. The step 8 failure
was driven by utility-loss mismatch, not by a fundamental property of
BCE utility.

## Step 10 — Statistical-threshold pruning of input weights in a 2-layer LTU network

**Goal**: steps 1–9 were all on a linear model, where every weight was
one hop from the output. Step 10 introduces a hidden layer and asks:
what utility should we assign to input → hidden weights — weights
that aren't directly connected to the output? We freeze each hidden
unit's input candidate pool to a random 256-of-1568 at init, train
the whole network with softmax CE, and apply step 6's statistical-
confidence threshold pruning only to the input → hidden weights.
Four candidate utilities are compared against a no-prune baseline.

### Architecture

- Input (1568) → **100 LTU hidden units** (hard step at 0, sigmoid-STE
  via [`ltu()` in `phd/jax_core/models.py`](../../../../jax_core/models.py))
  → 20 outputs with per-task softmax CE (same 2-task multi-MNIST as
  the rest of the progression).
- **Hidden → output is fixed 1-to-1**: unit `i` connects only to output
  `i // 5` (5 hidden units per output). The scalar on that connection
  is trainable. This gives each hidden unit a clean task identity,
  inherited from the output it drives.
- **Input → hidden fanin is fixed at init**: 256 of 1568 connections per
  hidden unit, sampled uniformly (≈128 same-task / ≈128 cross-task
  per unit). Only these 256 slots per unit are candidates for pruning.
- **W_in** is trained alongside W_out; gradients through W_in are
  masked by the current M_in.

### Utility variants

All variants share a preamble per step: (1) compute per-hidden-unit
flip utility under per-task softmax CE (`flip_utility_hidden`);
(2) derive binary LTU targets `target_h = where(U_flip < 0, 1 − h, h)`;
(3) compute the `informative_h = (h > 0) | (target_h > 0)` mask.

| Variant | Per-weight utility | K (leading factor) |
| ------- | ------------------ | ------------------ |
| `bce_ltu` | BCE remove utility per hidden unit against its binary `target_h`; no gating | 1 |
| `bce_ltu_informative` | Same as above, gated by `informative_h[:, None]` | 1 |
| `signed_ltu` | `\|e + c\| − \|e\|` with `e = target_h − σ(z1)`, `c = w·x`. Closest analog to step 3's signed. | 1 − 2/π |
| `signed_ltu_inf` | Target → ±∞ limit: `u = (2·target_h − 1) · w · x` (target=1 → +c; target=0 → −c) | 1 − 2/π |
| `no_prune` (reference) | — | — |

Threshold formula unchanged from step 6, K picked per variant as above.
Statistical-confidence CI fixed at 0.95.

### LR sweep (5 seeds, extended grid)

`{variant} × {2^-11 … 2^0}` (12 LRs) at SPP=50. All optima inside the
grid. Logged as `step10_lr_sweep_variant={v}_lr={lr}`.

| Variant | Best LR | Accuracy | Loss | Fanin F1 | Budget |
| ------- | ------- | -------- | ---- | -------- | ------ |
| `no_prune` | 0.25 | 0.664 | 1.090 | 0.665 | 25,600 |
| `bce_ltu` | 2^-5 (0.031) | 0.210 | 2.180 | 0.287 | 4,867 |
| `bce_ltu_informative` | 2^-4 (0.0625) | 0.160 | 2.243 | 0.295 | 4,740 |
| **`signed_ltu`** | 0.25 | **0.647** | 1.134 | 0.576 | 17,937 |
| `signed_ltu_inf` | 0.5 | 0.416 | 1.725 | 0.234 | 3,577 |

### Main sweep (20 seeds, 95% CI): variant × SPP

At each variant's best LR, SPP ∈ {50, 100, 200, 400}. 225k total steps
(n_cycles scales with SPP). Logged as `step10_main_variant={v}_spp={v}`.

Fanin F1 is per-hidden-unit precision/recall on the 256-candidate pool:
positive = kept, ground-truth positive = same-task input.

| Variant | SPP | Accuracy | Loss | Fanin F1 | Budget | Converge step |
| ------- | --- | -------- | ---- | -------- | ------ | ------------- |
| `no_prune` | 50 | 0.657 ± 0.005 | 1.105 ± 0.012 | 0.665 ± 0.001 | 25,600 | 0 |
| `no_prune` | 100 | 0.656 ± 0.006 | 1.111 ± 0.016 | 0.665 ± 0.001 | 25,600 | 0 |
| `no_prune` | 200 | 0.662 ± 0.006 | 1.093 ± 0.016 | 0.665 ± 0.001 | 25,600 | 0 |
| `no_prune` | 400 | 0.662 ± 0.006 | 1.093 ± 0.015 | 0.665 ± 0.001 | 25,600 | 0 |
| `bce_ltu` | 50 | 0.199 ± 0.015 | 2.185 ± 0.027 | 0.286 ± 0.007 | 4,766 ± 139 | 47,820 ± 2,655 |
| `bce_ltu` | 100 | 0.136 ± 0.006 | 2.278 ± 0.007 | 0.241 ± 0.005 | 3,967 ± 101 | 71,645 ± 3,542 |
| `bce_ltu` | 200 | 0.114 ± 0.003 | 2.301 ± 0.004 | 0.207 ± 0.006 | 3,456 ± 83  | 109,870 ± 5,545 |
| `bce_ltu` | 400 | 0.105 ± 0.001 | 2.308 ± 0.001 | 0.173 ± 0.003 | 2,977 ± 74  | 195,060 ± 8,943 |
| `bce_ltu_informative` | 50 | 0.161 ± 0.011 | 2.243 ± 0.016 | 0.298 ± 0.007 | 4,778 ± 126 | 31,882 ± 1,782 |
| `bce_ltu_informative` | 100 | 0.122 ± 0.002 | 2.293 ± 0.003 | 0.261 ± 0.005 | 4,097 ± 76 | 51,720 ± 2,389 |
| `bce_ltu_informative` | 200 | 0.111 ± 0.003 | 2.306 ± 0.004 | 0.231 ± 0.006 | 3,569 ± 97 | 85,130 ± 6,715 |
| `bce_ltu_informative` | 400 | 0.104 ± 0.001 | 2.315 ± 0.002 | 0.198 ± 0.004 | 3,045 ± 52 | 146,740 ± 8,236 |
| **`signed_ltu`** | 50  | 0.637 ± 0.005 | 1.156 ± 0.015 | 0.577 ± 0.003 | 18,002 ± 152 | 17,552 ± 1,440 |
| **`signed_ltu`** | 100 | 0.642 ± 0.005 | 1.145 ± 0.014 | 0.591 ± 0.002 | 19,295 ± 106 | 32,955 ± 1,786 |
| **`signed_ltu`** | 200 | 0.651 ± 0.005 | 1.115 ± 0.014 | 0.608 ± 0.001 | 20,839 ± 80  | 58,650 ± 6,109 |
| **`signed_ltu`** | 400 | **0.656 ± 0.004** | 1.107 ± 0.013 | **0.626 ± 0.002** | 22,365 ± 108 | 118,600 ± 13,030 |
| `signed_ltu_inf` | 50 | 0.422 ± 0.011 | 1.706 ± 0.027 | 0.240 ± 0.005 | 3,681 ± 82  | 28,670 ± 1,583 |
| `signed_ltu_inf` | 100 | 0.349 ± 0.010 | 1.879 ± 0.022 | 0.211 ± 0.003 | 3,236 ± 46  | 44,410 ± 1,439 |
| `signed_ltu_inf` | 200 | 0.315 ± 0.011 | 1.952 ± 0.026 | 0.198 ± 0.003 | 3,047 ± 46  | 63,200 ± 3,161 |
| `signed_ltu_inf` | 400 | 0.278 ± 0.009 | 2.038 ± 0.017 | 0.185 ± 0.003 | 2,814 ± 41  | 97,080 ± 4,779 |

### Interpretation

1. **`signed_ltu` at SPP=400 matches `no_prune`** on accuracy
   (0.656 ± 0.004 vs 0.662 ± 0.006) while pruning ~13% of the input
   connections, with fanin F1 = 0.626 (a random pruner on a 50/50
   pool would give F1 = 0.665 only by keeping everything; at this
   F1 with a non-trivial amount pruned, the pruner is selecting
   against cross-task connections). This is the headline positive
   result for step 10 — input-layer weights *can* be pruned with a
   useful utility, and `signed_ltu` is that utility.
2. **Step 5's SPP trend reverses here.** For the linear-model
   threshold pipeline, longer SPP reliably helped (step 5). For the
   2-layer BCE variants longer SPP *hurts* — accuracy drops monotonically
   from SPP=50 to SPP=400 for both `bce_ltu` (0.199 → 0.105) and
   `bce_ltu_informative` (0.161 → 0.104), and fanin F1 drops too. The
   BCE utility's signal doesn't get better with more warmup because the
   signal is wrong in direction — longer EMA just commits to a worse
   decision. `signed_ltu_inf` follows the same degenerate pattern
   (0.422 → 0.278).
3. **For `signed_ltu` the SPP trend matches step 5's** — longer SPP
   prunes less aggressively and leaves behind a more task-selective
   mask (F1 rises 0.577 → 0.626 as SPP grows, budget rises
   18k → 22k, accuracy rises 0.637 → 0.656). Same shape as step 5's
   signed-utility behavior, just transported to a 2-layer setup.
4. **BCE variants are anti-selective.** Fanin F1 never exceeds 0.30
   for `bce_ltu`/`bce_ltu_informative`, meaning the pruner is dropping
   same-task connections at a *higher* rate than cross-task. The
   `informative` mask doesn't rescue this; its numbers are within noise
   of vanilla `bce_ltu`. Compare to step 9: BCE utility under softmax-
   CE training also underperformed there, but in the fixed-budget
   dynamic-prune paradigm it at least tracked chance-level alignment
   (~0.52) instead of going below it. The statistical-threshold variant
   in step 10 amplifies the mistake because bad decisions accumulate
   in the frozen post-convergence state.
5. **`signed_ltu_inf` loses information relative to `signed_ltu`.**
   Replacing `σ(z1)` with the binary-target sign (the ±∞-target limit)
   collapses accuracy from 0.656 to 0.422 at their respective best
   SPPs, and F1 from 0.626 to 0.240. The finite-target version uses a
   continuous prediction that encodes how confident the LTU already
   is in its target; the ±∞ version throws that away and rates every
   contribution only by whether it points toward the target sign.
6. **Convergence step correlates strongly with SPP × budget**. SPP=400
   runs take 90k–200k steps to converge (approaching the 225k total),
   so those configs have very little post-convergence eval window.
   Accuracy / F1 measured during active pruning for those; same caveat
   as step 5's SPP=800 and step 6's SPP=400.

**Bottom line**: `signed_ltu` successfully extends step 6's statistical-
threshold pruning to the input layer of a 2-layer LTU network, matching
the no-prune accuracy while prefentially removing cross-task
connections. BCE utilities do not work in this setting — not because
statistical threshold pruning is wrong for a hidden layer, but because
the BCE per-unit signal has no reliable direction-of-loss information
for an LTU with a binary target derived from a downstream flip utility.

## Scripts


| Script                        | Purpose                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `common.py`                   | Shared utilities — fixed mask samplers, purity/entropy/alignment, MLflow helpers |
| `01_linear_global.py`         | Steps 1–3 core: linear model, global pruning, configurable utility + budget      |
| `02_budget_sweep.py`          | Step 2: budget sweep at contribution utility                                     |
| `03_signed_utility.py`        | Step 3: signed LOO utility across same budget sweep                              |
| `04_threshold_pruning.py`     | Step 4: LR sweep for threshold pruning from fully connected                      |
| `05_spp_sweep.py`             | Step 5: SPP sweep for threshold pruning                                          |
| `06_lr_sweep.py`              | Step 6: LR sweep for statistical-threshold pruning on normalized MNIST           |
| `06_statistical_threshold.py` | Step 6: CI × SPP sweep for statistical-confidence threshold pruning              |
| `07_lr_sweep.py`              | Step 7: LR sweep for demand-driven generation                                    |
| `07_generation.py`            | Step 7: allocation-method sweep (clipped_linear vs softmax)                      |
| `08_lr_sweep.py`              | Step 8: LR sweep for statistical-threshold pruning with BCE utility              |
| `08_statistical_threshold.py` | Step 8: CI × SPP sweep for BCE-utility statistical-threshold pruning             |
| `09_lr_sweep.py`              | Step 9: LR × budget × variant sweep for the utility comparison                   |
| `09_main.py`                  | Step 9: 20-seed main run at per-(variant, budget) best LRs                       |
| `10_lr_sweep.py`              | Step 10: variant × LR sweep for 2-layer LTU input-weight pruning                 |
| `10_main.py`                  | Step 10: 20-seed main run, variant × SPP at per-variant best LR                  |


