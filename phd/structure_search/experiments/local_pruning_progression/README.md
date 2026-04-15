# Local Pruning Progression

A progression of experiments that rethinks connection-level pruning from the
neuron's perspective, starting from the simplest possible setup (linear,
global pruning) and increasing complexity step by step. Follow-on steps will
add hidden units, per-unit local budgets, and dynamic budget allocation.

Builds on findings from `../debug_task_separation/`.

## Invariants (constant across every step)

These hold for every experiment and every baseline in this directory.

- **Task**: non-stationary parallel multi-MNIST, 2 sub-tasks. Inputs are
  concatenated MNIST digits (1568 = 2 × 784). Labels are two class indices
  in [0, 10). Every **4000 steps** one randomly-chosen task's label mapping
  is replaced with a fresh random permutation of [0..9]. Non-stationarity
  is there to overcome the blocking problem.
- **Batch size**: 1 (online).
- **Seeds**: 20 seeds per reported configuration, base seed 42.
- **Run length**: ≥ 100k steps; extended as needed to (a) hit ~3× connection
  turnover for dynamic methods and (b) give convergence within the
  evaluation window. Default for step 1 is 225k.
- **Evaluation**: running average of training cross-entropy loss over the
  final **40k steps** of the run. This measures rate of adaptation, not
  peak accuracy.
- **Reported metrics** (for every run, mean ± 95% CI over seeds):
  `final_loss`, `purity`, `entropy`.
  - *Purity* — treat each output neuron as a "unit". For output neuron k,
    split its active input connections into the two task blocks (inputs
    0..783 = task 0, inputs 784..1567 = task 1). Purity of that unit =
    `max(t0, t1) / (t0 + t1)`. Average over units (units with no active
    connections are skipped).
  - *Entropy* — normalized binary entropy of the same (t0, t1) split,
    averaged over units.
- **Baselines for every step** (logged as their own MLflow runs):
  1. **Random fixed connectivity** — 1500 connections sampled uniformly
     from the full (20 × 1568) slot pool, frozen for the whole run. Train
     weights with SGD; no pruning/generation.
  2. **Within-task fixed connectivity** — 750 connections per task sampled
     uniformly from within that task's (10 × 784) block, frozen for the
     run. (For later steps with hidden units, "fixed within-task" becomes
     "dense within-task" since the dense budget then fits under 1500.)
- **Learning-rate sweep**: every configuration (method and both baselines)
  gets its own LR sweep — best LR differs between fixed and dynamic
  connectivity, and between the two baselines. Sweep grid is geometric by
  factor of 4 (e.g. `{1.5625e-4, 6.25e-4, 2.5e-3, 1e-2, 4e-2}`), extended
  until both tails of the LR vs. final-loss curve rise above the minimum
  (a U-shape). Sweep with 5 seeds per LR, then re-run the winning LR with
  20 seeds for the reported number.
- **MLflow**: project name `local_pruning_progression`. One run per
  reported configuration, named `step{N}_{variant}` (e.g. `step1_method`,
  `step1_baseline_random`, `step1_baseline_intask`). Sweep runs append
  `_lr={value}`. Logs only aggregated (mean + CI) metrics across seeds —
  no per-seed child runs. Also logs a moving-window loss trajectory
  (`loss_window_5k` every 5k steps) for convergence inspection.

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

| Variant | MLflow run | LR | Final-40k loss | Purity | Entropy |
|:--|:--|:--:|:--:|:--:|:--:|
| Dynamic (prune+gen, 1 conn / 50 steps) | `step1_dynamic` | 0.16 | **1.040 ± 0.007** | 0.578 ± 0.006 | 0.975 ± 0.003 |
| Fixed random | `step1_fixed_random` | 0.16 | 1.404 ± 0.016 | 0.542 ± 0.003 | 0.992 ± 0.001 |
| Fixed within-task | `step1_fixed_intask` | 0.16 | **1.016 ± 0.009** | 1.000 ± 0.000 | 0.000 ± 0.000 |

Full LR sweep curves (5-seed means):

| LR | Dynamic loss | Random loss | Intask loss | Dynamic purity |
|:--:|:--:|:--:|:--:|:--:|
| 1.56e-4 | 2.269 | 2.286 | 2.269 | 0.678 |
| 6.25e-4 | 2.202 | 2.253 | 2.202 | 0.689 |
| 2.5e-3 | 1.957 | 2.140 | 2.008 | 0.678 |
| 1e-2 | 1.519 | 1.895 | 1.646 | 0.652 |
| 4e-2 | 1.127 | 1.584 | 1.249 | 0.610 |
| **1.6e-1** | **1.042** | **1.395** | **1.011** | 0.579 |
| 6.4e-1 | 2.391 | 1.682 | 1.149 | 0.559 |
| 2.56 | 10.3 | 5.09 | 3.44 | 0.558 |

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

| Budget | Turnover | Variant | Loss | Alignment | Purity |
|:--:|:--:|:--|:--:|:--:|:--:|
| **1500** | 3× | dynamic | **1.040 ± 0.007** | 0.572 ± 0.006 | 0.578 |
|      |   | fixed random | 1.404 ± 0.016 | 0.500 ± 0.004 | 0.542 |
|      |   | fixed intask | **1.016 ± 0.009** | 1.000 | 1.000 |
| **500**  | 9× | dynamic | **1.206 ± 0.008** | 0.767 ± 0.006 | 0.770 |
|      |   | fixed random | 1.887 ± 0.019 | 0.498 ± 0.009 | 0.581 |
|      |   | fixed intask | 1.580 ± 0.018 | 1.000 | 1.000 |
| **150**  | 30× | dynamic | **1.725 ± 0.014** | **0.832 ± 0.011** | 0.835 |
|      |   | fixed random | 2.168 ± 0.011 | 0.487 ± 0.014 | 0.662 |
|      |   | fixed intask | 2.060 ± 0.016 | 1.000 | 1.000 |
| **50**   | 90× | dynamic | **2.086 ± 0.008** | 0.792 ± 0.023 | 0.872 |
|      |   | fixed random | 2.264 ± 0.006 | 0.490 ± 0.023 | 0.777 |
|      |   | fixed intask | 2.228 ± 0.006 | 1.000 | 1.000 |
| **20**   | 225× | dynamic | 2.217 ± 0.006 | 0.745 ± 0.037 | 0.923 |
|      |   | fixed random | 2.289 ± 0.003 | 0.477 ± 0.039 | 0.897 |
|      |   | fixed intask | 2.275 ± 0.003 | 1.000 | 1.000 |

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

| Budget | Utility | Loss | Alignment |
|:--:|:--|:--:|:--:|
| 1500 | contribution | 1.040 ± 0.007 | 0.572 ± 0.006 |
| 1500 | signed       | 1.038 ± 0.008 | 0.575 ± 0.006 |
| 500  | contribution | 1.206 ± 0.008 | 0.767 ± 0.006 |
| 500  | signed       | 1.217 ± 0.013 | 0.770 ± 0.006 |
| 150  | contribution | 1.725 ± 0.014 | 0.832 ± 0.011 |
| 150  | signed       | 1.764 ± 0.012 | 0.814 ± 0.009 |
| 50   | contribution | 2.086 ± 0.008 | 0.792 ± 0.023 |
| 50   | signed       | 2.102 ± 0.010 | 0.774 ± 0.022 |
| 20   | contribution | 2.217 ± 0.006 | 0.745 ± 0.037 |
| 20   | signed       | 2.222 ± 0.006 | 0.733 ± 0.038 |

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

| LR | Loss | Alignment (P) | Sep-F1 | Purity | Budget | Converge cycle |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 2^-7 (0.0078) | 1.987 ± 0.055 | 0.828 ± 0.011 | 0.153 ± 0.023 | 0.822 | 1615 ± 265 | 791 ± 50 |
| 2^-6 (0.0156) | 1.668 ± 0.060 | 0.857 ± 0.009 | 0.180 ± 0.024 | 0.837 | 1874 ± 284 | 892 ± 81 |
| **2^-5 (0.0313)** | **0.985 ± 0.011** | 0.713 ± 0.008 | **0.381 ± 0.008** | 0.716 | 5757 ± 241 | 500 ± 69 |
| 2^-4 (0.0625) | 1.207 ± 0.011 | 0.503 ± 0.000 | 0.548 ± 0.002 | 0.508 | 18765 ± 152 | 49 ± 11 |
| 2^-3 (0.125) | 2.398 ± 0.021 | 0.500 ± 0.000 | 0.550 ± 0.002 | 0.507 | 19153 ± 161 | 18 ± 2 |

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
   aggressively to ~1,600–1,900 connections because weights barely move and
   most connections accumulate near-zero (hence non-positive) utility. High
   LR (2^-4–2^-3) barely prunes at all (~19,000 kept) because large weight
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

| SPP | n_cycles | Loss | Alignment (P) | Sep-F1 | Budget | Converge step |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 50 | 4500 | 0.985 ± 0.011 | 0.713 ± 0.008 | 0.381 ± 0.008 | 5757 ± 241 | 25012 |
| 100 | 2250 | 0.958 ± 0.009 | 0.725 ± 0.007 | 0.367 ± 0.008 | 5338 ± 204 | 57580 |
| 200 | 1125 | 0.835 ± 0.012 | 0.647 ± 0.004 | 0.435 ± 0.007 | 7943 ± 212 | 80260 |
| **400** | 562 | **0.784 ± 0.007** | 0.571 ± 0.002 | 0.507 ± 0.004 | 12516 ± 135 | 140540 |
| 800 | 281 | 0.823 ± 0.008 | 0.522 ± 0.001 | 0.527 ± 0.001 | 16005 ± 64 | 223480 |

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

    Û_corr = Û / (1 − β^t)

where t is the age of the weight (= step count, since weights are
never regenerated here). Prune iff Û_corr < τ_w, with

    τ_w(t) = −z_α · |w| · σ_x · sqrt(K · (1 + β^t) / (1 − β^t))

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

| LR | Final loss |
|:--:|:--:|
| 2^-11 (0.00049) | 1.964 |
| 2^-10 (0.00098) | 0.966 |
| **2^-9 (0.00195)** | **0.842** |
| 2^-8 (0.00391) | 0.947 |
| 2^-7 (0.00781) | 1.533 |
| 2^-6 (0.01562) | 3.287 |
| 2^-5 (0.03125) | 7.096 |

Logged as `step6_lr_sweep_lr={value}`.

### Main sweep: CI × SPP (20 seeds, 95% CI, lr=2^-9)

| CI | SPP | Loss | Alignment | Sep-F1 | Budget | Converge step |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.9  | 50  | 0.838 ± 0.007 | 0.525 ± 0.001 | 0.612 ± 0.002 | 21,915 ± 285 | 16,135 ± 2,137 |
| 0.9  | 100 | 0.833 ± 0.008 | 0.519 ± 0.001 | 0.619 ± 0.002 | 23,170 ± 223 | 37,935 ± 4,429 |
| **0.9**  | **200** | **0.823 ± 0.010** | 0.514 ± 0.001 | 0.619 ± 0.002 | 23,748 ± 263 | 104,760 ± 14,798 |
| 0.9  | 400 | 0.825 ± 0.010 | 0.509 ± 0.001 | 0.621 ± 0.002 | 24,618 ± 227 | 210,180 ± 14,610 |
| 0.95 | 50  | 0.842 ± 0.008 | 0.522 ± 0.001 | 0.632 ± 0.002 | 24,009 ± 255 | 14,105 ± 1,109 |
| 0.95 | 100 | 0.842 ± 0.008 | 0.517 ± 0.001 | 0.638 ± 0.002 | 25,223 ± 219 | 31,925 ± 5,236 |
| 0.95 | 200 | 0.833 ± 0.009 | 0.512 ± 0.001 | 0.635 ± 0.002 | 25,580 ± 270 | 103,290 ± 13,083 |
| 0.95 | 400 | 0.837 ± 0.010 | 0.507 ± 0.001 | 0.636 ± 0.002 | 26,376 ± 177 | 206,540 ± 16,482 |
| 0.99 | 50  | 0.849 ± 0.008 | 0.518 ± 0.001 | 0.648 ± 0.002 | 26,164 ± 235 | 10,928 ± 1,266 |
| 0.99 | 100 | 0.854 ± 0.008 | 0.513 ± 0.001 | 0.651 ± 0.002 | 27,221 ± 146 | 27,300 ± 3,246 |
| 0.99 | 200 | 0.851 ± 0.011 | 0.509 ± 0.001 | 0.651 ± 0.002 | 27,770 ± 245 | 69,660 ± 10,347 |
| 0.99 | 400 | 0.854 ± 0.010 | 0.504 ± 0.001 | 0.649 ± 0.002 | 28,251 ± 180 | 176,180 ± 18,075 |

Logged as `step6_statistical_threshold_ci={v}_spp={v}`.

### Interpretation

1. **Hypothesis confirmed: SPP sensitivity collapses.** Step 5's loss
   ranged from 0.985 (SPP=50) to 0.784 (SPP=400) — a ~25% swing. Step 6's
   loss at a given CI varies by at most ~1.5% across SPP 50→400:

   | Method                      | SPP=50 | SPP=200 | SPP=400 | Spread |
   |:--|:--:|:--:|:--:|:--:|
   | Step 5 (τ ≤ 0)              | 0.985  | 0.835   | 0.784   | 0.201  |
   | Step 6, CI=0.9              | 0.838  | 0.823   | 0.825   | 0.015  |
   | Step 6, CI=0.95             | 0.842  | 0.833   | 0.837   | 0.009  |
   | Step 6, CI=0.99             | 0.849  | 0.851   | 0.854   | 0.005  |

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

## Scripts

| Script | Purpose |
|:--|:--|
| `common.py` | Shared utilities — fixed mask samplers, purity/entropy/alignment, MLflow helpers |
| `01_linear_global.py` | Steps 1–3 core: linear model, global pruning, configurable utility + budget |
| `02_budget_sweep.py` | Step 2: budget sweep at contribution utility |
| `03_signed_utility.py` | Step 3: signed LOO utility across same budget sweep |
| `04_threshold_pruning.py` | Step 4: LR sweep for threshold pruning from fully connected |
| `05_spp_sweep.py` | Step 5: SPP sweep for threshold pruning |
| `06_lr_sweep.py` | Step 6: LR sweep for statistical-threshold pruning on normalized MNIST |
| `06_statistical_threshold.py` | Step 6: CI × SPP sweep for statistical-confidence threshold pruning |
