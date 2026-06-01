# Handoff: Learned backward path-gates for credit assignment — MultiMNIST experiments

**For:** a Claude Code agent running the experiments.
**Goal:** test whether learning a per-weight gate on the *backward* pass (IDBD-style
meta-learning) improves performance on a multi-task, non-stationary problem
(MultiMNIST), relative to standard backprop.

This package ships a **verified** math core (`path_gates.py`), a training script
(`train_multimnist.py`), a sweep config, and this document. The risky part — the
custom backward pass and the closed-form meta-update — has already been checked
against autodiff and finite differences (see §4). Your job is to wire it into the
lab infrastructure, run the sweep, and analyze the results.

---

## 1. The idea

Standard backprop sends the signal `delta_j * w_ji` backward across edge `i -> j`.
We insert a per-weight multiplicative **gate** `g_ji = sigmoid(beta_ji)` on the
**backward path only**, and *learn* the gates `beta` online by gradient descent on
the loss — exactly the IDBD move (Sutton 1992), but with the meta-parameter being a
backward-path gate rather than a step-size.

Why this might help non-stationary multi-task learning: in a shared trunk, different
tasks need credit routed differently, and credit that is correct for one task can be
interference for another. A learned backward gate can attenuate edges that carry
misleading credit, i.e. it learns a *structural* credit-assignment mask. The
hypothesis is that this reduces cross-task interference and improves online accuracy
and/or retention under task switching.

This is an instance of **MetaOptimize** (Sharifnassab, Salehkaleybar, Sutton, ICML
2025) with the meta-parameter chosen to be the backward gates; IDBD is the same
framework with the meta-parameter being log step-size.

---

## 2. Formulas

Notation: weight `w_ji` on forward edge `i -> j` (node `i` lower, `j` higher);
`net_i` is the pre-activation of node `i`, `a_i = f(net_i)`; `f'` the activation
derivative; `alpha` base step-size; `theta` meta step-size.

**Modulated backward signal** (what the base learner trains on):
```
s_i = f'(net_i) * sum_j  sigmoid(beta_ji) * w_ji * s_j
```
At the output, `s` is seeded with the loss gradient (softmax−onehot for CE).

**Base weight update** (plain SGD on the modulated signal):
```
w_ik  <-  w_ik  -  alpha * s_i * a_k
```

**True backward signal** (gates = 1; the ordinary backprop delta):
```
delta_i_true = f'(net_i) * sum_j  w_ji * delta_j_true
```

**Meta-gradient for the gates** (leading order — see derivation note below):
```
dL/dbeta_ji  ~=  - sigma'(beta_ji) * [ f'(net_i) * w_ji * s_j ] * delta_i_true * ||a_in||^2
                  \_____________/   \________ c_ji __________/   \__________/
                    gate slope         this edge's backward push    true error at i
```
`c_ji` is how much this edge contributes to the (modulated) signal at node `i`;
`delta_i_true` is the true error that *should* arrive at `i`. The rule **opens** a
gate when the edge's backward push aligns with the true error, and **closes** it when
it carries misleading credit. `||a_in||^2` is the input energy of the weights that
`s_i` updates (the IDBD `x^2` normalizer); it is a per-sample scalar and only rescales
per-layer magnitude. Gate update: `beta <- beta - theta * dL/dbeta`.

**Layer/matrix form** (as implemented, layer `l` mapping `a^(l) -> net^(l)`, gate
`beta^(l)`): with `s_out = s` at `net^(l)`, `delta_in` the true delta at `net^(l-1)`,
`fprime_in = f'(net^(l-1))`,
```
dL/dbeta^(l) = - sigma'(beta^(l)) ⊙ W^(l) ⊙ ( s_out^T @ (fprime_in ⊙ delta_in) ) / B * ||a^(l-1)||^2
```

**Two structural facts** (both encoded in the code, worth knowing):
- The **input-layer gate is inert**: it modulates credit flowing back to the network
  inputs, which update no weights. Its meta-grad is identically zero — don't learn it.
- The closed form is **exact at batch size 1** (the fully-online regime) and is a
  *diagonal-in-batch* approximation for `B > 1` (it drops cross-sample input-Gram
  coupling). See §4.

**Derivation note / approximation.** The exact meta-gradient (MetaOptimize `H`-matrix
recursion) tracks how every weight depends on every gate over time. The rule above is
the **leading-order, local** term: it keeps each gate's direct effect on the weights
its own signal updates and drops diffuse higher-order paths — the same spirit as
Sutton's `dw_j/dbeta_i ~= 0` approximation that turns the IDBD trace local. `gamma = 0`
recovers hypergradient descent (Baydin); `gamma > 0` is the discounted-future-loss
version, realized cheaply here as an EMA (eligibility trace) on the meta-gradient.

---

## 3. Variants to test

| name | gates | meta-update | extra cost | rationale |
|------|-------|-------------|-----------|-----------|
| `baseline` | open, frozen (σ(8)≈1) | none | — | **control = standard backprop** |
| `fixed` | half-open, frozen (σ(0)=0.5) | none | — | ablation: does *attenuation alone* matter, vs *learning*? |
| `honest` | learned | `delta_true` from a 2nd unmodulated backward, `gamma=0` | +1 backward pass | principled version |
| `proxy` | learned | `delta_true ≈ s` (reuse modulated signal), `gamma=0` | none | cheap; no 2nd backward |
| `honest_trace` | learned | `honest` + EMA trace (`gamma=0.9`) | +1 backward pass | temporal credit for **tracking** the non-stationarity |

**Most promising to watch:** `honest` and `honest_trace`. The eligibility-trace
version is the natural fit for the non-stationary regime (it can adapt the gates as
the active task switches). `proxy` is the interesting cheap option — if it matches
`honest`, the second backward pass is unnecessary. `baseline` and `fixed` are the
controls that make any positive result meaningful.

All five are selectable via `--variant`; see the table in `train_multimnist.py`.

---

## 4. The verified core (`path_gates.py`)

Pure `jax`/`numpy`, manual forward + two manual backward passes + closed-form
meta-update. `Params` and `MetaState` are registered pytrees, so `step` jits cleanly.
Key functions: `init_params`, `forward`, `softmax_ce_seed`, `backward(..., gated=)`,
`meta_grad(..., use_proxy=)`, `step(...)`.

Run the checks yourself first:
```
python verify_core.py        # uses float64
```
Expected output (already confirmed on this machine):
```
[Test 1] manual backward vs autodiff   max|Δ| = 8.11e-17   PASS
[Test 2] single-hidden meta-grad: cosine(analytic, finite-diff) = 1.0000  sign-agreement = 100.00%   PASS
[Test 3] deep-net leading-order quality: trunk cosine ≈ +0.89, head cosine ≈ +0.58, sign-agreement ≈ 82%
CORE OK
```
Interpretation:
- **Test 1** — with gates=1 the manual backward reproduces autodiff exactly. The
  forward/backward plumbing is correct.
- **Test 2** — at **batch size 1** the closed-form meta-gradient matches finite
  differences exactly (cosine 1.0). The formula is right in the online regime.
- **Test 3** — on a deeper net with all gates active and `B>1`, the leading-order +
  diagonal-in-batch approximation gives a correct *descent direction* (cosine ~0.9 on
  the trunk; weaker on heads), not the exact gradient. This is expected and fine for
  optimization. Batch-size degradation (separately measured): cosine ≈ 1.000 (B=1),
  0.998 (B=2), 0.978 (B=8), 0.916 (B=64). **Include `batch_size=1` in the sweep** — it
  is both the exact regime and the natural streaming setting.

---

## 5. Experiment: non-stationary multi-task MultiMNIST

**Data.** Side-by-side MNIST pairs: place two random digits next to each other →
`28×56 = 1568`-dim input, with two labels `(left, right)`. (Loader in
`train_multimnist.py`; it tries `keras` / `torchvision` / a local `mnist.npz`.)

**Model.** Shared MLP trunk `1568 → 256 → 256` (ReLU) + two linear heads `256 → 10`
(left-digit, right-digit). Plain SGD for both the base weights and the gates — this
matches the derivation (MetaOptimize's IDBD special case is SGD base + SGD meta).

**Non-stationarity (primary).** A periodic **hard task switch**: only one head's loss
is active in each block of `switch_every` steps (default 2000), alternating
left/right. This is the IDBD sign/relevance-flip analogue — it forces the shared trunk
to be reused under shifting credit-assignment demands, creating interference and
forgetting pressure. The gates (a single, slowly-learned mask) must find a routing
that reduces *average* interference; the `honest_trace` variant can additionally
*track* the switches over time.

**Metrics** (prequential / online — predict before each update):
- `online_acc_active` — accuracy on the currently-active task (primary objective).
- `retention_acc_inactive` — accuracy on the task *not* being trained (forgetting).
- `gate_mean_trunk`, `gate_mean_head` — diagnostics: do gates move and develop
  structure? (std growth, per-layer means).
- Summary scalars `final_online_acc`, `final_retention_acc` for the sweep objective.

**Alternative non-stationarity (optional, if you want a second axis).** Keep *both*
heads active (joint loss) and apply a fixed input pixel-permutation that changes every
`K` steps (permuted-MultiMNIST). Sustained multi-task + drift rather than hard
switching. Easy to add as a `--schedule {switch,permute}` flag.

---

## 6. Files & how to run

```
path_gates.py        verified math core (don't change the math; extend if needed)
verify_core.py       correctness checks — run first
train_multimnist.py  experiment: data + non-stationary loop + variants + MLflow
sweep_multimnist.yaml mlflow-sweeper config
run_sweep.sh         zero-friction bash grid (no Hydra needed)
HANDOFF.md           this file
```

**Smoke test (already passes on synthetic data here):**
```
python verify_core.py
MLFLOW_TRACKING_URI=file:./mlruns python train_multimnist.py --variant honest \
    --num_steps 2000 --switch_every 500 --hidden 64 64 --seed 0
```

**Single real run:**
```
export MLFLOW_TRACKING_URI=...     # lab tracking server (Nginx/token auth per your setup)
python train_multimnist.py --variant honest_trace --alpha 0.0625 --theta 64 \
    --batch_size 1 --num_steps 40000 --switch_every 2000 --seed 0
```

**Sweep — pick one path:**
- *Quick:* `bash run_sweep.sh` (grid over the argparse flags; logs to MLflow).
- *Integrated (lab `mlflow-sweeper`):* use `sweep_multimnist.yaml`. **Caveat:**
  `train_multimnist.py` uses `argparse` (`--name value`) but `mlflow-sweeper` appends
  Hydra-style `name=value`. Reconcile by either (a) refactoring `parse_args()` to read
  an OmegaConf config so overrides apply, or (b) using the programmatic `run_sweep()`
  API (sweep skill → `mlflow_config.md`, "Run Programmatically") with a thin function
  that maps the config dict onto `main()`'s logic. The `jax-experiment` lab template
  also uses argparse, so whatever convention you pick, keep it consistent with that.

**Compute Canada:** the full grid (5 variants × αs × θs × batch × seeds) is modest;
it runs locally on the RTX 4080. Use CC only if you expand the grid substantially —
see the sweep skill's `cc_deployment.md`.

**Analysis:** use the lab `experiment-analysis` skill to pull the MLflow runs and
scaffold notebooks. Key figures: (1) online accuracy vs step with task-switch
boundaries marked, per variant; (2) retention vs step; (3) sensitivity of
`final_online_acc` to `theta`, hued by variant; (4) gate-structure diagnostics
(mean/std of `sigmoid(beta)` over training, and a heatmap of a trunk gate matrix at
the end).

---

## 7. Practical notes & gotchas

- **`theta` must be swept wide and high.** The meta-gradient is a product of several
  small quantities, so gates move slowly; useful `theta` is typically `>> alpha`
  (the config sweeps up to `2^9 = 512`). If gates barely move (`std(sigmoid(beta))`
  stays ~0) at the top of the range, push higher, or RMS-normalize the per-layer
  meta-grad before stepping (a practical knob that makes `theta` interpretable, at the
  cost of departing from the clean IDBD scaling). `baseline`/`fixed` ignore `theta`.
- **`beta_init`.** Learned variants start at `beta=2.0` (`sigmoid≈0.88`, mostly open)
  so training begins close to backprop and gates only deviate where it helps. Worth a
  small sweep (e.g. `{0, 2, 4}`).
- **Batch size.** `B=1` is exact and the natural streaming regime; larger `B` is the
  diagonal-in-batch approximation. Compare both.
- **Performance.** `train_multimnist.py` does a separate `forward` for the
  prequential metric and again inside `step`; fine for correctness, but if you need
  speed, fuse them (return logits from `step`) and/or wrap the inner loop in
  `jax.lax.scan` per task-block (the `jax-experiment` template's pattern) — note the
  block boundary is where `task_w` changes.
- **Don't "fix" Test 3 to be exact.** The <1.0 cosine there is the documented
  leading-order/diagonal-in-batch approximation, not a bug.

---

## 8. What success looks like

A positive result: `honest` and/or `honest_trace` match or beat `baseline` on
`online_acc_active` **and/or** improve `retention_acc_inactive`, most plausibly at
small batch and around task-switch boundaries, with `fixed` *not* explaining the gain
(so it's the *learning*, not mere attenuation). Gates should develop non-trivial
structure (std > 0, sensible per-layer pattern). If `proxy` ≈ `honest`, the second
backward pass is unnecessary — a useful finding on its own.

A negative result is also informative: if learned gates don't help here, it bounds
where backward-path modulation matters and tells you whether to (a) move to harder
interference (more tasks, faster switching, deeper trunk), (b) make gates
input-conditional rather than static parameters, or (c) implement the exact
MetaOptimize `H`-recursion instead of the leading-order rule.

---

## 9. Extension (if the leading-order rule underperforms)

The exact non-diagonal meta-gradient is the MetaOptimize `H`-matrix recursion
(`H_tau = (1-gamma) sum gamma^... dw/dbeta`, meta-grad `= H^T grad f`). It's heavier
(tracks weight-gate sensitivities) but removes both the locality and the
diagonal-in-batch approximations. The 2×2 / Hessian-free approximations in that paper
are the practical middle ground. Flag this back if the local rule plateaus; the core
is structured so the meta-update can be swapped without touching the forward/backward.

---

*Source grounding:* IDBD — Sutton (1992), `incompleteideas.net/papers/sutton-92a.pdf`.
MetaOptimize — Sharifnassab, Salehkaleybar, Sutton (ICML 2025), `arxiv.org/abs/2402.02342`.
