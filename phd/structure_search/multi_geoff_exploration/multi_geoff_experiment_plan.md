# Multi-GEOFF Structure Learning — Experiment Plan

**Status:** all open questions resolved; ready to hand off for implementation. A few minor defaults are marked *(assumed)* inline — flag if wrong.

## Goal and Strategy

The long-term goal is an algorithm that learns the connectivity structure of a neural network online. The strategy is to start from a setup that is very likely to work because it encodes privileged information about the problem, then transition step by step toward an algorithm that still works while encoding progressively less privileged information.

## Background and Code Pointers

Context for whoever implements this (human or LLM):

- **Motivating paper:** *Connectivity, Credit Assignment, and the Speed of Learning* (Meyer, Freeman & Sutton, RLC 2026 CRL workshop). Key findings: in multi-task, non-stationary problems (multi-MNIST), dense connectivity misassigns credit across unrelated tasks, which slows learning and cannot be fixed by scaling; block-sparse connectivity (one independent subnetwork per task) is the reference good structure; connectivity also governs the largest step-size a network can tolerate (effective step-size argument, Section 3.2 of the paper).
- **Paper code:** `phd/structure_search/` (JAX/equinox, Hydra config in `conf/config.yaml`). Contains the existing multi-MNIST baselines: dense, block-sparse, random sparse, and SET. Dense establishes a performance lower bound, block-sparse close to an upper bound. `block_sparse_mlp.py` has `BlockSparseMLP` and `compute_hidden_dim_for_params` (hit a parameter budget by adjusting hidden count).
- **The new problem — multi-GEOFF:** `phd/jax_core/tasks/multi_geoff.py` (`MultiGEOFFTask`). A synthetic, regression version of multi-MNIST: the target is `n_tasks` independent GEOFF sub-networks ("slots") with block-diagonal teacher connectivity; per-output squared-error loss. Non-stationarity permutes the columns of one random slot's readout matrix every `perturb_period` steps — the analog of multi-MNIST's label permutation. Helpers: `perturbation_period(per_task_period, n_tasks)` converts a per-task period to a global one; `fraction_variance_explained` and `irreducible_mse` for scale-free evaluation.
- **Why multi-GEOFF instead of multi-MNIST:** Autostep is derived specifically for regression (squared error) and does not work for classification objectives. Consequence: none of the existing multi-MNIST baselines carry over, so new baselines are needed (see the Baselines Interlude below).
- **Autostep:** `phd/jax_core/optimizers/idbd.py`, `optax_idbd(autostep=True, ...)`. Takes `(loss_grads, prediction_grads)` as its update input. `version='prediction_grads'` (the default) is the drop-the-hessian approximation (Variant A below). The current code targets a single linear layer (asserts no biases, warns when given multiple weight matrices) — proper two-layer support is part of the work here.
- **Related notebooks:** step-size diagnostics for IDBD/Autostep already exist in `phd/sandbox/feature_sifting/algorithm_testing/exploration/step_size_analysis.ipynb` and may be useful reference for the plotting in Experiment 1.
- Experiments 1 and 2 go in separate notebooks.

### Common Experimental Settings

- **Non-stationarity everywhere:** every experiment uses the non-stationary problem, with the same per-task permutation rate throughout — each task's readout is permuted on average once every 2,000 steps. Via `perturbation_period(2000, n_tasks)`: a global permutation every 1,000 steps for the 2-task problems (Experiment 1) and every 250 steps for the 8-task problems (baselines, Experiment 2).
- **Task dimensions:** multi-GEOFF defaults (20 input features, 10 outputs per task, `noise_std=1`) unless noted.
- **Optimizer:** Autostep for everything, baselines included.
- **Initial step-size:** deliberately small and fixed at ~1e-5 for all experiments and all connections (including connections added mid-training). No sweeping of the initial step-size anywhere.
- **Meta step-size:** one reliable global value, used everywhere. Try a few values to settle on it; around 0.005 is expected to work well. Higher might perform slightly better, but too high diverges — the hard requirement is that no runs diverge, so prefer the reliable value.

### Two Autostep Variants for Networks

Autostep is derived for the linear case. Extending it to neural networks requires approximating the update for *h* to avoid a hessian term. Experiment 1 tests two approximations, and all Experiment 1 plots should be produced for both:

- **Variant A (exists):** drop the hessian term entirely; the normalizer uses squared prediction gradients. This is the existing `version='prediction_grads'` mode.
- **Variant B (to add):** treat each layer as if it were its own independent linear problem. Wherever linear Autostep uses the squared input x², use the squared value of the weight's source unit: actual network inputs for first-layer weights, hidden-unit values for second-layer weights. This mode needs to be added to the Autostep code.

Everything after Experiment 1 uses Variant A (`prediction_grads`); there is only a small chance this changes after seeing the Experiment 1 results.

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

Counts are sampled uniformly at random per hidden unit *(assumed)*. New weights are initialized to zero *(assumed, as in the paper's dense-transition)*, with step-sizes at the standard initial value (~1e-5).

Because each affected hidden unit now reads some of the other task's inputs *and* feeds some of its outputs, the added connections are bad but not strictly useless — there is some incentive for their weights to be nonzero. The best solution should nevertheless keep them all at 0 with small step-sizes.

**Plots:** same as 1a, restricted to steps after the connectivity change.

> **Possible follow-up:** if some added connections' step-sizes rise by more than is comfortable, ablate how the number of added connections affects the rise (or whether it is independent of that number).

## Baselines Interlude — Dense and Block-Sparse on Multi-GEOFF

Because the problem changed from multi-MNIST, there are no existing baseline numbers. Before Experiment 2, run dense and block-sparse baselines on the 8-task non-stationary multi-GEOFF problem (common settings):

- Sweep each method over multiple parameter counts (varying hidden count to hit each budget, per the `compute_hidden_dim_for_params` convention).
- All configurations use the common Autostep settings (fixed ~1e-5 initial step-size, the chosen meta step-size). The "try a few meta step-sizes" reliability check can be folded into these runs.
- Plot performance vs. parameter count for both methods together. Performance = mean fraction of variance explained over the last 10% of steps, matching the paper's evaluation convention *(assumed)*.

These curves serve two purposes: they replace the paper's Figure 4 numbers as the reference points, and they determine Experiment 2's connection budget — roughly the smallest parameter count at which block-sparse performs well. Experiment 2 uses that many parameters and no more.

## Experiment 2 — Learn a Good Solution with Online Connectivity Changes

The goal now shifts from analysis to actually learning a good solution: separate subnetworks and low loss. Use more subtasks so the gap between good and bad solutions stands out more.

**Setup:** 8-task multi-GEOFF; 64 total hidden units; connection budget taken from the baselines interlude (approximately what block-sparse needs to do well, and no more); permutations per common settings (global period 250). Optimizer: Autostep, Variant A (`prediction_grads`).

**Initialization:**

1. Connect each hidden unit to a single output unit, chosen at random. These are the only hidden→output connections at initialization.
2. Divide the remaining connection budget evenly across all output + hidden units. Connect each hidden unit to that many random input units, and each output unit directly to that many random input units (input→output skip connections). This preserves the rule that each hidden unit has exactly one connection to the output layer.

### 2a — Static Initial Structure

Train with the initial architecture held fixed. This is the baseline for Experiment 2, and it is expected to perform poorly.

**Measure:** MSE, fraction of variance explained (`fraction_variance_explained`), connectivity separation, signal separation, and the mean number of incoming and outgoing connections per hidden unit.

**Metric definitions:**

- **Connectivity separation (0–1):** starting from an output unit, trace every backward path along connections toward the input layer (a direct input→output skip connection counts as a length-1 path). The metric is the fraction of those paths that terminate at an input belonging to the same subtask as the output. The same input can be counted multiple times — the count is over unique paths from the output unit to input units. Average over all output units.
- **Signal separation (0–1):** the same metric, but each path's count is weighted by the absolute value of the product of the weights along the path. Path weights are normalized per output unit so they sum to 1, then the metric is averaged over all output units.

### 2b — Online Connectivity Changes

Begin changing the connectivity of the network online.

**Gradient masking:** to prevent cross-task connections from contaminating hidden-unit gradients, mask all gradients flowing from output units back into hidden units, except through the hidden→output connections created at initialization. Newly added hidden→output connections still receive their own weight updates, but are stop-gradiented on the backward pass — no error flows through them into the hidden unit and the layers below. This has no effect on learning until new hidden→output connections start being added. Correspondingly, the initial 1:1 hidden→output connections are exempt from pruning.

**Pruning:** start conservative — prune a connection only when it is believed to be completely useless. Criterion: **TBD (placeholder, to be filled in).**

**Generation:** when and how to add new connections: **TBD (placeholder).**
