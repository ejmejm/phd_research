# Connection Slots, earlier notebooks

Notebooks 3 to 6 import `../methods.py`. `single_change/` holds the first versions of notebooks 4 and 6, with one output sign change, and the `methods.py` they used.

## Current experiments

- `3_bounded_search.ipynb` tests repeated search against fixed pools, all inputs at the same budget, and ideal connections.
- `4_changed_weights.ipynb` reverses one output repeatedly, compares continued search with fixed connections, and looks inside the networks.
- `5_promotion_step.ipynb` changes only the promotion step, or only the initial step for ordinary connections without candidates.
- `6_recovery_cost.ipynb` separates connection changes, unit replacement, and active connection removal under the same reversals.
- `methods.py` holds the shared code. Each experiment uses five seeds and output noise of standard deviation 0.1.
- `old/` holds the earlier versions of notebooks 4 and 6 and of `methods.py`, with one output sign change at example 50,000.

Notebooks 3 and 5 use 100,000 examples on a fixed task.
The target has 108 inputs, four LTUs with four inputs each, and four outputs with two LTUs each.
The proposed methods and both baselines use 2,244 slots, biases included. The ideal reference uses 32 slots.
All connection checks occur every 1,000 examples. Every 10,000 examples, search attempts to replace one weak unit per group of eight.
Parent choices respect available output slots. Source draws inspect only each unit's own slots.

Fixed pools fail on this problem. Separate candidates approach the ideal reference. Small initial steps learn more slowly.
Small steps at promotion remove the early advantage of separate candidates.
Large steps without candidate protection give worse final accuracy, so a larger step alone does not fix that method.
Both proposed methods use Autostep. These results do not establish a general limit of step-size adaptation.

Notebooks 4 and 6 use a target with the same inputs and four LTUs, but two outputs that each read all four LTUs.
All target weights are +1 or −1. Every 50,000 examples, all weights of the first output reverse, four times in 250,000 examples.
The second output and all hidden weights stay fixed. Fixed-connection copies stop structure changes at the first reversal.
All comparisons share targets and examples. Plots show mean RMSE without uncertainty bands.

After a reversal, the output weights take their new signs fast, and error decreases.
Then hidden LTUs whose shadow is on the reversed output lose their features, and error rises again.
In the ideal reference, the shadow place causes these losses. LTUs whose shadow is on the fixed output keep their features.
With fixed connections, most of the reversed output's weight stays on LTUs with such shadows, and every reversal repeats the losses.
Connection changes move that weight to LTUs whose shadow is on the fixed output, and later recoveries become fast.
Removal of active connections is not necessary for this. Unit replacement alone gives the slowest recovery.
These results do not yet establish all the success conditions.

## Earlier experiments

**Goal.** GROW in `../pruning_and_depth/` keeps a gradient sum for every input that an LTU does not read, so the state
of each LTU grows with the number of inputs N. Here each LTU has K slots, each one a connection or a candidate (an
input whose gradient sum it keeps). With M LTUs of K slots, the method should do about as well as the current method
(K = N) with M·K/N LTUs: the same state, but a cost per LTU that does not grow with N.

Each notebook asks one question, on the target network of `../pruning_and_depth/1_connection_budget.ipynb` (16 target
LTUs, each reading 10 of 1000 inputs), with 160 LTUs of 100 slots, at output noise of std 0.1 and 1:

1. `1_fixed_slots.ipynb` — can LTUs with 100 slots, filled at birth, do as well as the current method? Yes, if the
   candidates are drawn by the parent's weights: they hold most of the right inputs. Drawn uniformly, they learn about
   as little as LTUs with no candidates.
2. `2_freeing_candidates.ipynb` — with uniform candidates, can LTUs that free the candidates whose z-score is low do as
   well? Yes, at z_low = 0.5, 1 and 2: level with the current method with 160 LTUs. A higher z_low learns faster at
   first.

## The method

Lines tagged with a notebook number are new here; the rest is as in `../pruning_and_depth/` (one layer).

```
setup     a fixed number of LTUs, each with K slots; a slot holds one input: a connection or a candidate       (1)

every step
  LEARN   network Autostep, as before; for each slot j of LTU u: G[u, j] += g, S[u, j] += g², g = dL/dz_u · x_j  (1)

every 1k steps
  GROW    each LTU u: z = |G| / sqrt(S) over its candidates; if max z > 6, u reads that candidate (w 0, α 0.03),
          and G[u, :] = S[u, :] = 0                                                                             (1)
          each candidate with z < z_low takes a new input, drawn uniformly from the inputs in none of u's slots,
          with G = S = 0                                                                                        (2)

every 10k steps
  PRUNE   cut connections whose weight changed sign since the last PRUNE; a cut input stays as a candidate      (1)
  SEARCH  replace the weakest 1 in 8 LTUs by NEW LTUs                                                           (1)

NEW LTU u
  parent p and shadow weight as before; its first 20 slots are connections to inputs drawn ∝ w[p, j]², signed as
  before; its other K − 20 slots are candidates, drawn uniformly (or ∝ w[p, j]², in notebook 1)                 (1)
```

With K = N, every input is in a slot, and this is the current method.

## Plan and pilot (2026-11-17)

**Method, as in `../pruning_and_depth/1_connection_budget.ipynb` (one layer), except:**
- a fixed number of LTUs, in place of the connection budget; search replaces the weakest 1 in 8 every 10k steps;
- each LTU has K slots, connections plus candidates; a connection that the sign rule cuts becomes a candidate;
- a new LTU draws K inputs ∝ its parent's linear weights²: the first 20 are connections, the rest candidates;
- GROW (every 1k steps) connects the candidate with the largest |z| if |z| > 6, with z = G / sqrt(Σ g²), the
  candidate's own score (g = dL/dz · x_j, G = Σ g since it entered or since the LTU last grew). With K = N, this is
  about the old test |G| > 6 · rms(G), as all candidates have the same age (checked once).

**Curves** (RMSE, last 100k steps, noise std 0.1 and 1, 5 seeds, 500k steps, N = 1000, K = 100):
- bar: the current method (K = N) with 16 LTUs, the same state as the proposed method;
- ceiling: the current method with 160 LTUs;
- naive baseline: 160 LTUs, K = 100 slots drawn at birth and never changed except by the sign rule's cuts;
- proposed (test rule): 160 LTUs, K = 100; at each GROW check, first free each candidate with |z| < Z_LOW and draw a
  new input ∝ the parent's linear weights² (among inputs not in the slots). Z_LOW ∈ {0.5, 1, 2}. A wrong candidate's
  z is about N(0, 1) at any age, so Z_LOW = 1 frees about 2/3 of the wrong candidates per check.

**Step 1, calibration.** Targets: (A) that of `1_connection_budget` (16 target LTUs reading 10 of 1000 inputs, each
of 10 outputs reads 4); (B) 64 target LTUs, each output reads 16 (160 right inputs per output, more than K).
- C1: keep a target only if, at noise 0.1, the naive baseline's excess RMSE over the noise std is at least twice the
  ceiling's. Prediction: A fails (a pool of 100 drawn by the parent's linear weights holds nearly all 40 right inputs
  of the parent), B passes.

**Step 2, the test rule** on the kept target.
- H1: with Z_LOW = 1, last-100k RMSE ≤ 1.05 × the bar's, at both noise levels.
- H2: with Z_LOW = 1, at noise 0.1, it closes at least half of the gap from the naive baseline to the ceiling.
- If the test rule fails H1 or H2 for every Z_LOW: the CBP rule (free the candidate with the lowest utility trace
  at a fixed rate, after a maturity age).

**Step 1 results** (pilot `slots.py`, scratchpad; RMSE over the last 100k steps, noise 0.1 / 1):
- A: bar 0.175 / 1.034, ceiling 0.150 / 1.026, naive 0.152 / 1.028. C1 fails, as predicted: the parent's draw puts
  most right inputs in the slots, and 160 LTUs give many draws. The z test equals the old rms test (0.152 / 1.025).
- B: bar 0.613 / 1.178, ceiling 0.559 / 1.181, naive 0.607 / 1.181. C1 fails because nothing learns in time: the
  ceiling leaves the linear learner's level only after ~300k steps, and is still falling at 500k.
- Extra check on A: with no candidates (K = 20, the first connections only) 0.505 / 1.143, so growth matters. With
  candidates drawn uniformly (the first 20 connections still by the parent), naive 0.483 / 1.136: C1 passes
  (excess 0.383 vs 0.050).

**Change for step 2:** target A, and candidates drawn uniformly (at a new LTU and at each refill); a new LTU's first 20
connections are still drawn by its parent's weights. Bars from step 1: H1 needs ≤ 0.184 / 1.086; H2 needs ≤ 0.317 at
noise 0.1.

**Step 2 results** (same pilot, 30 runs in 314 s): the test rule ends at 0.151 / 1.027 for every Z_LOW in
{0.5, 1, 2}, level with the ceiling (0.150 / 1.026). H1 passes (bar 0.175 / 1.034). H2 passes, as it closes nearly
all of the gap from the naive baseline (0.483). Z_LOW changes only the early speed: at 100k steps (noise 0.1) 0.298,
0.198 and 0.183 for Z_LOW 0.5, 1 and 2, against the ceiling's 0.177. A higher Z_LOW (more turnover) is faster.

**Diagnostics, fixed in advance.** At the end of each run, for each LTU, its best-matched target LTU (the one whose
inputs it reads most): how many of its 10 inputs the LTU reads, and how many are in its slots.
