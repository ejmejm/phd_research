# Multi-GEOFF Structure Learning — Experiment Plan

**Status:** updated after Experiment 1 was run — its outcomes (Autostep Variant B, L1 regularization) are folded in. Experiment 2 (stage validation) now runs before Experiment 3 (online connectivity changes), since the maturity rule should be validated before it is used. Minor defaults are marked *(assumed)* inline — flag if wrong.

## Goal and Strategy

The long-term goal is an algorithm that learns the connectivity structure of a neural network online. The strategy is to start from a setup that is very likely to work because it encodes privileged information about the problem, then transition step by step toward an algorithm that still works while encoding progressively less privileged information.

## Background and Code Pointers

Context for whoever implements this (human or LLM):

- **Motivating paper:** *Connectivity, Credit Assignment, and the Speed of Learning* (Meyer, Freeman & Sutton, RLC 2026 CRL workshop). Key findings: in multi-task, non-stationary problems (multi-MNIST), dense connectivity misassigns credit across unrelated tasks, which slows learning and cannot be fixed by scaling; block-sparse connectivity (one independent subnetwork per task) is the reference good structure; connectivity also governs the largest step-size a network can tolerate (effective step-size argument, Section 3.2 of the paper).
- **Paper code:** `phd/structure_search/` (JAX/equinox, Hydra config in `conf/config.yaml`). Contains the existing multi-MNIST baselines: dense, block-sparse, random sparse, and SET. Dense establishes a performance lower bound, block-sparse close to an upper bound. `block_sparse_mlp.py` has `BlockSparseMLP` and `compute_hidden_dim_for_params` (hit a parameter budget by adjusting hidden count).
- **The new problem — multi-GEOFF:** `phd/jax_core/tasks/multi_geoff.py` (`MultiGEOFFTask`). A synthetic, regression version of multi-MNIST: the target is `n_tasks` independent GEOFF sub-networks ("slots") with block-diagonal teacher connectivity; per-output squared-error loss. Non-stationarity permutes the columns of one random slot's readout matrix every `perturb_period` steps — the analog of multi-MNIST's label permutation. Helpers: `perturbation_period(per_task_period, n_tasks)` converts a per-task period to a global one; `fraction_variance_explained` and `irreducible_mse` for scale-free evaluation.
- **Why multi-GEOFF instead of multi-MNIST:** Autostep is derived specifically for regression (squared error) and does not work for classification objectives. Consequence: none of the existing multi-MNIST baselines carry over, so new baselines are needed (see the Baselines Interlude below).
- **Autostep:** `phd/jax_core/optimizers/idbd.py`, `optax_idbd(autostep=True, ...)`. Takes `(loss_grads, prediction_grads)` as its update input. `version='prediction_grads'` (the default) is the drop-the-hessian approximation (Variant A below). The current code targets a single linear layer (asserts no biases, warns when given multiple weight matrices) — proper two-layer support is part of the work here.
- **Related notebooks:** step-size diagnostics for IDBD/Autostep already exist in `phd/sandbox/feature_sifting/algorithm_testing/exploration/step_size_analysis.ipynb` and may be useful reference for the plotting in Experiment 1. Experiment 1's own notebook (`multi_geoff_exploration/experiment_1.ipynb`) contains the L1 implementations to match.
- Experiments 1, 2, and 3 each go in their own notebook.

### Common Experimental Settings

- **Non-stationarity everywhere:** every experiment uses the non-stationary problem, with the same per-task permutation rate throughout — each task's readout is permuted on average once every 2,000 steps. Via `perturbation_period(2000, n_tasks)`: a global permutation every 1,000 steps for the 2-task problems (Experiments 1 and 2) and every 250 steps for the 8-task problems (baselines, Experiment 3).
- **Task dimensions:** multi-GEOFF defaults (20 input features, 10 outputs per task, `noise_std=1`) unless noted.
- **Optimizer:** Autostep for everything, baselines included.
- **Initial step-size:** deliberately small and fixed at 1e-6 for all experiments and all connections — initial networks, connections added mid-training (1b), reset connections (Experiment 2), and generated connections (Experiment 3). No sweeping of the initial step-size anywhere.
- **Meta step-size:** one reliable global value, used everywhere. Try a few values to settle on it; around 0.005 is expected to work well. Higher might perform slightly better, but too high diverges — the hard requirement is that no runs diverge, so prefer the reliable value.

### Two Autostep Variants for Networks

Autostep is derived for the linear case. Extending it to neural networks requires approximating the update for *h* to avoid a hessian term. Experiment 1 tests two approximations, and all Experiment 1 plots should be produced for both:

- **Variant A (exists):** drop the hessian term entirely; the normalizer uses squared prediction gradients. This is the existing `version='prediction_grads'` mode.
- **Variant B (to add):** treat each layer as if it were its own independent linear problem. Wherever linear Autostep uses the squared input x², use the squared value of the weight's source unit: actual network inputs for first-layer weights, hidden-unit values for second-layer weights. This mode needs to be added to the Autostep code.

**Outcome — Experiment 1 has now been run:** Variant B is the one to keep. Everything after Experiment 1 uses squared source-unit values as the x² term wherever such a term appears — the h update, and the v trace and prune test introduced in Experiments 2 and 3.

## Experiment 1 — Can Autostep Separate Useful from Useless Connections?

The planned method will only work if step-size adaptation works well, so start by verifying that Autostep can separate relevant from irrelevant inputs/hidden units in a network where relevance is known by construction.

### 1a — Mixed-Relevance Network Trained from Scratch

**Problem:** 2-task multi-GEOFF (non-stationary per common settings).

**Network:** 64 hidden units, block-sparse base structure — each hidden unit fully connected to its own task's inputs and outputs. Because each hidden unit's connections then belong to a single subtask, any added cross-task connection is known to be useless. Modify the base structure as follows:

- ⅓ of hidden units: additionally fully connected to the other task's outputs.
- ⅓ of hidden units: additionally fully connected to the other task's inputs.
- ⅓ of hidden units: unchanged.

**Procedure:** train from scratch with Autostep (both variants).

**Plots** (per Autostep variant):

1. Step-sizes vs. time step for a subsample of all weights — useful connections in blue, useless in red — split into separate panels for input-layer and output-layer weights.
2. The same plot but for weight values instead of step-sizes.

The real quantity of interest is the distribution over the useful and useless groups over time; subsampled individual traces are the starting point for seeing it.

> **Possible follow-up (noted, not scheduled):** start from a fully good structure and add an irrelevant feature mid-training, with vs. without Autostep. How much does it hinder learning? Does Autostep ever erroneously raise the step-size, and how frequently? What if many irrelevant features are added at once?

### 1b — Bad Connections Added Mid-Training

**Question:** starting from a good structure (pure block-sparse), letting the network learn for a while, and only then adding new bad connections — do those connections' step-sizes and weight magnitudes rise, or stay down?

**Procedure:** same problem and network scale as 1a, starting from the unmodified 64-hidden block-sparse structure. Train for a fixed number of steps, chosen to be long enough for the *level* of performance to stabilize (weights never stop changing under non-stationarity, but the performance level recovered after each permutation should be steady). Then, for **each** hidden unit, add both:

- 1–3 new outgoing connections to the other task's outputs, and
- 1–10 new incoming connections from the other task's inputs.

Counts are sampled uniformly at random per hidden unit *(assumed)*. New weights are initialized to zero *(assumed, as in the paper's dense-transition)*, with step-sizes at the standard initial value (1e-6).

Because each affected hidden unit now reads some of the other task's inputs *and* feeds some of its outputs, the added connections are bad but not strictly useless — there is some incentive for their weights to be nonzero. The best solution should nevertheless keep them all at 0 with small step-sizes.

**Plots:** same as 1a, restricted to steps after the connectivity change.

> **Possible follow-up:** if some added connections' step-sizes rise by more than is comfortable, ablate how the number of added connections affects the rise (or whether it is independent of that number).

## Experiment 2 — Validating the Connection Maturity Stages

Experiment 3's final variant restricts where new connections may come from, using a notion of unit "maturity" built from per-connection stages. That maturity rule is a first idea and may not be a good one, so validate it here first — visually and quantitatively, on a problem where good and bad connectivity is known by construction.

### Stage Definitions

- A connection is **nascent** when it is created. It enters a **growth** phase when its weight crosses the jitter threshold. It becomes **mature** when it is in the growth state and its h value crosses zero.
- The **jitter threshold** for a weight is the point at which it is not clearly distinguishable from the weight of a completely useless connection jittering around 0. Each hidden and output unit maintains a trace v of its squared error over its incoming weights (defined in the pseudocode below), from which an h jitter threshold is computed: h is jitter-sized when hᵢ² < αᵢ v / 4, i.e., |hᵢ| < √(αᵢ v)/2. (This same test later serves as the "h is small" part of Experiment 3's prune rule.) The w jitter threshold is 2× the h jitter threshold, with two differences: the v used for it is bias-corrected, and the growth stage is unreachable until the v trace is at least 50% occupied (bias-correction division term > 0.5) — otherwise v would start at 0 and every connection would immediately enter the growth stage.
- A hidden unit's stage is the latest stage among its outgoing connections, so it matures when its first outgoing connection matures. All input units are mature from the first step.

**Stage-tracking pseudocode** — run after the normal training update, for each hidden and output unit u, with incoming weights w, source values x, per-unit error δᵤ (an output unit's own prediction error; for a hidden unit, its backpropagated error signal *(assumed)*), and per-unit traces v and c:

```
v ← v + (1/k) Σᵢ αᵢ xᵢ² (δᵤ² − v)      # squared-error trace, k = 5 (constant)
c ← c + (1/k) Σᵢ αᵢ xᵢ² (1 − c)        # trace occupancy (bias-correction term)

for each incoming connection i:
    J_h = √(αᵢ v) / 2                  # h jitter threshold (raw v)
    J_w = √(αᵢ v / c)                  # w jitter threshold = 2·J_h with bias-corrected v
    if stageᵢ = nascent and c > 0.5 and |wᵢ| > J_w:
        stageᵢ ← growth
    else if stageᵢ = growth and h_prev,i · hᵢ < 0:     # h crossed zero
        stageᵢ ← mature                                # absorbing until reset/pruned

unit stage = most advanced stage among u's outgoing connections
```

### Experiment Design

**Setup:** reuse Experiment 1a's mixed-relevance construction — 2-task multi-GEOFF, 64 hidden units, block-sparse base, ⅓ of hidden units additionally fully connected to the other task's outputs, ⅓ to its inputs, ⅓ unchanged — so every connection has a ground-truth useful/useless label. Initialize exactly as in Experiment 1 (weights cannot all start at 0, or hidden units would output 0 and no gradients would flow).

**Training:** train as in Experiment 1 (Autostep, Variant B), plus the same L1 regularization used everywhere from here on (directly as part of the gradient, coefficient 0.01). **No pruning and no generation** — just compute the v and c traces and the stages as training runs.

**Reset protocol:** at the halfway point of training, once things are somewhat stable, begin occasionally resetting randomly chosen connections: weight → 0, step-size → 1e-6, h → 0 *(assumed)*, stage → nascent. A reset connection looks exactly like a freshly generated one, but with a known usefulness label. The reset schedule (how often, how many) is at the implementer's discretion — enough resets of both useful and useless connections to get statistics.

**Visual checks:**

1. Stage raster plot: a subsample of connections on the y-axis, grouped into useful and useless; time on the x-axis; color = stage. Expected picture after resets: useful connections progress nascent → growth → mature; useless connections mostly stay nascent.
2. For a small sample of reset useful and useless connections, overlay |w| against J_w and h against ±J_h, marking stage-transition times.

**Quantitative checks** (over reset connections):

1. Fraction of useless connections that ever enter the growth stage, compared against the fraction expected by chance under the jitter model (a purely jittering weight crossing 2× the h jitter threshold).
2. Fraction of useless connections that reach maturity (hopefully near 0) vs. fraction of useful connections that reach maturity (hopefully around all).
3. Distribution of time from reset to maturity for useful connections.

If the connection-level statistics look right, unit-level maturity follows directly from them; if they don't, revise the thresholds or stage definitions before running Experiment 3's Variant 2.

## Baselines Interlude — Dense and Block-Sparse on Multi-GEOFF

Because the problem changed from multi-MNIST, there are no existing baseline numbers. Before Experiment 3, run dense and block-sparse baselines on the 8-task non-stationary multi-GEOFF problem (common settings):

- Sweep each method over multiple parameter counts (varying hidden count to hit each budget, per the `compute_hidden_dim_for_params` convention).
- All configurations use the common Autostep settings (fixed 1e-6 initial step-size, the chosen meta step-size). The "try a few meta step-sizes" reliability check can be folded into these runs.
- Plot performance vs. parameter count for both methods together. Performance = mean fraction of variance explained over the last 10% of steps, matching the paper's evaluation convention *(assumed)*.

These curves serve two purposes: they replace the paper's Figure 4 numbers as the reference points, and they determine Experiment 3's connection budget — roughly the smallest parameter count at which block-sparse performs well. Experiment 3 uses that many parameters and no more.

## Experiment 3 — Learn a Good Solution with Online Connectivity Changes

The goal now shifts from analysis to actually learning a good solution: separate subnetworks and low loss. Use more subtasks so the gap between good and bad solutions stands out more.

**Setup:** 8-task multi-GEOFF; 64 total hidden units; connection budget taken from the baselines interlude (approximately what block-sparse needs to do well, and no more); permutations per common settings (global period 250). Optimizer: Autostep, Variant B (squared source-unit values).

**L1 regularization:** all Experiment 3 runs apply an L1 regularization term directly as part of the gradient (the subgradient form, not the proximal form), on all connection weights, with coefficient 0.01 — set by a rule of thumb from the problem parameters and found extremely useful in Experiment 1. This coefficient is not a hyperparameter to sweep over.

**Initialization:**

1. Connect each hidden unit to a single output unit, chosen at random. These are the only hidden→output connections at initialization.
2. Divide the remaining connection budget evenly across all output + hidden units. Connect each hidden unit to that many random input units, and each output unit directly to that many random input units (input→output skip connections). This preserves the rule that each hidden unit has exactly one connection to the output layer.

### 3a — Static Initial Structure

Train with the initial architecture held fixed. This is the baseline for Experiment 3, and it is expected to perform poorly.

**Measure:** MSE, fraction of variance explained (`fraction_variance_explained`), connectivity separation, signal separation, and the mean number of incoming and outgoing connections per hidden unit.

**Metric definitions:**

- **Connectivity separation (0–1):** starting from an output unit, trace every backward path along connections toward the input layer (a direct input→output skip connection counts as a length-1 path). The metric is the fraction of those paths that terminate at an input belonging to the same subtask as the output. The same input can be counted multiple times — the count is over unique paths from the output unit to input units. Average over all output units.
- **Signal separation (0–1):** the same metric, but each path's count is weighted by the absolute value of the product of the weights along the path. Path weights are normalized per output unit so they sum to 1, then the metric is averaged over all output units.

### 3b — Online Connectivity Changes

Begin changing the connectivity of the network online. Connections are pruned when judged useless and replaced one-for-one elsewhere, so the total connection count stays at the budget.

**Gradient masking (base version):** to prevent cross-task connections from contaminating hidden-unit gradients, mask all gradients flowing from output units back into hidden units, except through the hidden→output connections created at initialization. Newly added hidden→output connections still receive their own weight updates, but are stop-gradiented on the backward pass — no error flows through them into the hidden unit and the layers below. This has no effect on learning until new hidden→output connections start being added. Correspondingly, the initial 1:1 hidden→output connections are exempt from pruning.

**Pruning.** The prune rule adapts the following linear learning algorithm (LMS with Autostep and feature pruning) to each unit of the network:

```
for each step:
    δ = y − wᵀx                          # prediction error
    α ← AutostepUpdate(α, h, δ, x)
    w_prev ← w
    w ← w + α ⊙ δx                       # LMS update
    h ← h ⊙ (1 − α ⊙ x²) + α ⊙ δx
    v ← v + (1/k) Σᵢ αᵢ xᵢ² (δ² − v)      # MSE trace
    for each feature i:
        if w_prev,i · wᵢ < 0             # weight crossed zero
           and hᵢ² < αᵢ v / 4:           # h is small
            wᵢ ← 0;  hᵢ ← 0;  αᵢ ← α_init
            regenerate feature i         # new random feature takes the slot
```

With a single output, pruning a feature is the same as pruning a connection. In the network, this algorithm — specifically the v trace and the prune test — is applied to each hidden and output unit over its incoming weights:

- One v trace per hidden and output unit; k is a constant, fixed at 5.
- Every x² term is the squared value of the connection's source unit — actual squared inputs for hidden units, squared hidden-unit values for connections from hidden units to outputs — i.e., the Variant B convention, used in the h update and everywhere else x² appears above.
- A connection is pruned when its weight crossed zero on this step and its h is small (h² < αᵢ v / 4). Pruning zeroes its weight and h, and the freed capacity goes to the generation rule below ("regenerate" simply means a new connection is created, not necessarily on the same unit).

**Generation.** When a connection is pruned, create one new connection: select a destination uniformly at random from all output and hidden units (all get an equal chance), then select a source uniformly at random from the units in layers prior to the destination (input units for a hidden unit; input and hidden units for an output unit), sampling only among pathways where a connection does not already exist. The new connection's weight starts at 0, and its step-size at the standard initial value, 1e-6.

#### 3b Variants

Run the base version first, then two variants in sequence:

**Variant 1 — no special initial connections.** The network still starts from the exact same initialization as 3a — including exactly one outgoing connection per hidden unit — but those initial hidden→output connections are no longer special: they can be pruned like any other connection, and all connections carry gradients (the gradient masking above is removed).

**Variant 2 — maturity-gated generation.** Keeps Variant 1's changes and adds one more: the source of a newly generated connection must be a **mature** unit, per Experiment 2's stage definitions (all input units are mature from the first step; a hidden unit matures when its first outgoing connection matures). The full per-step algorithm combines the two pieces of pseudocode already given — 3b's prune rule and Experiment 2's stage tracking:

```
for each step:
    forward pass; compute per-output errors; backpropagate per-unit errors δᵤ
    for each hidden and output unit u (incoming weights w, source values x):
        α ← AutostepUpdate(α, h, δᵤ, x)         # Variant B: x² = squared source values
        w_prev ← w
        w ← w + α ⊙ (δᵤ x − λ sign(w))          # LMS update with L1 (λ = 0.01)
        h ← h ⊙ (1 − α ⊙ x²) + α ⊙ δᵤ x
        v ← v + (1/k) Σᵢ αᵢ xᵢ² (δᵤ² − v)        # k = 5
        c ← c + (1/k) Σᵢ αᵢ xᵢ² (1 − c)          # occupancy (bias correction)
        for each incoming connection i:
            # stage update
            if stageᵢ = nascent and c > 0.5 and |wᵢ| > √(αᵢ v / c):
                stageᵢ ← growth
            else if stageᵢ = growth and h_prev,i · hᵢ < 0:
                stageᵢ ← mature
            # prune test
            if w_prev,i · wᵢ < 0 and hᵢ² < αᵢ v / 4:
                remove connection i
                dest ← uniform random over all hidden and output units
                src  ← uniform random over MATURE units in layers prior to dest,
                       among pathways with no existing connection
                create src → dest with w = 0, h = 0, α = 1e-6, stage = nascent
```

Gradient masking is off in this variant, so backpropagation is standard. For exactly where the L1 term enters (weight update vs. h update), match the existing Experiment 1 notebook implementation.