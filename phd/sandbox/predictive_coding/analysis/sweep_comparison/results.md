# Rotating MNIST Sweep Results

## Overview

We ran 12 hyperparameter sweeps (96 total runs) testing streaming iPC on the Rotating MNIST task: a single MNIST digit rotates by δ degrees per step, and a new digit is drawn after each full rotation. The network predicts the digit class from the flattened 28×28 image, with online accuracy measured *before* each weight update.

Five algorithm variants were compared at their individually tuned learning rates:
- **Streaming iPC** — value nodes persist across observations, never reinitialized
- **Forward-init iPC** — value nodes reinitialized via top-down forward pass each step
- **EMA iPC** — blend of persistent and forward-init value nodes (β=0.9)
- **BP (streaming)** — standard backprop on the same temporally correlated data stream
- **BP (shuffled)** — standard backprop on i.i.d. shuffled data

Default architecture: 3 layers, 256 hidden, ReLU, δ=5°, T=1.


## Key Results

### Best asymptotic accuracy per method (δ=5°)

| Method | Best Accuracy | Best α / LR |
|---|---|---|
| Streaming iPC | 98.76% | 2⁻¹² |
| EMA iPC (β=0.9) | 98.76% | 2⁻¹² |
| Forward-init iPC | 97.25% | 2⁻⁶ |
| BP (shuffled) | 97.32% | 2⁻⁴ |
| BP (streaming) | 97.18% | 2⁻⁴ |

### The repeat-previous-label baseline

A trivial strategy: predict whatever the label was on the previous step. Since the digit identity is constant during a full rotation (360/δ steps), this baseline is wrong only once per digit switch — the first step after a new digit is drawn.

| δ (deg/step) | Steps per digit | Baseline accuracy |
|---|---|---|
| 1° | 360 | 99.72% |
| 5° | 72 | 98.61% |
| 15° | 24 | 95.83% |

At δ=5°, this baseline achieves **98.61%** — higher than all methods except streaming and EMA iPC.


## Interpretation

### What the results show

**Streaming iPC works mechanically at T=1.** The persistent value nodes don't cause divergence, instability, or degraded learning. All sweeps converge, energy decreases over training, and accuracy reaches a stable plateau. The approach is viable.

**The optimal learning rate differs dramatically by variant.** Streaming iPC needs α ≈ 2⁻¹², while forward-init needs α ≈ 2⁻⁶ — a 64× difference. Forward-init with α=2⁻⁴ causes energy to explode (>10¹⁵). This suggests the persistent value nodes produce smoother, lower-variance error signals that require smaller weight updates. It also means you cannot smoothly interpolate between streaming and forward-init (via EMA β) without also adjusting α.

**γ=0.5 is confirmed as the robust inference learning rate.** γ=0.1 is too conservative (accuracy drops 2-3pp), γ=1.0 overshoots (drops 1-4pp depending on α). The interaction with α is notable: high γ requires low α for stability.

**T=1 is sufficient for accuracy in this regime.** T ∈ {1, 2, 4, 8} all achieve approximately the same accuracy (~98.76%) at optimal α. The difference is in energy: T=8 achieves ~14× lower energy than T=1 at the same α. Accuracy saturates before energy converges.

**Architecture doesn't matter at this scale.** Depth (3/4/5 layers) and width (256/512) give nearly identical accuracy. The task is solved by the smallest architecture tested.

### What the results don't show

**We cannot confidently rank methods against each other.** Without confidence intervals, the differences between streaming iPC (98.76%), BP shuffled (97.32%), and the trivial baseline (98.61%) are not clearly significant. All methods with reasonable hyperparameters land in a narrow band between ~97% and ~99%. The general trend — streaming iPC ≥ EMA iPC > forward-init ≈ BP — is suggestive but not conclusive.

**We cannot conclude that the hidden layers are doing useful work.** The classification label is constant for 72 consecutive steps (at δ=5°). During this time, the "correct" strategy is to maintain the same prediction regardless of input. Streaming iPC's persistent value nodes are naturally inertial — they carry forward with minimal change when the input changes slowly. This inertia *looks* like high accuracy, but it may just be stickiness. The output layer could learn to maintain its prediction without the hidden layers contributing meaningful input-dependent computation.

**The high trivial baseline means the signal-to-noise ratio is poor.** With 98.61% achievable by doing nothing, the remaining 1.4% of headroom is where all the interesting variation lives. This makes it difficult to distinguish genuine learning from inertia, or to detect meaningful differences between methods.

### The energy analysis is the most informative result

Unlike accuracy, energy is not confounded by the trivial baseline. The free energy F = Σ||ε^(l)||² measures how well the network's internal generative model fits the data — it reflects the quality of the internal representation, not just the output.

Key energy findings:
- Energy decreases substantially over training, confirming the network is learning (not just being inertial)
- Higher T reduces energy exponentially (~3× per doubling of T) while accuracy stays flat — the internal model improves even when the output doesn't
- Larger α reduces energy (better fit) until it causes instability
- Forward-init at α=2⁻⁴ causes energy explosion while streaming iPC at the same α is stable


## Per-Sweep Details

### Phase 1: Learning rate sensitivity

Streaming and EMA iPC show a broad plateau of good performance across α ∈ {2⁻¹⁴, 2⁻¹², 2⁻¹⁰}, with degradation only at the extremes. Forward-init has a narrower stable range and explodes at moderate α. BP variants show a clean bowl shape peaking around 2⁻⁴.

### T sweep (inference steps per observation)

At optimal α (2⁻¹⁴ or 2⁻¹²), all T values give ~98.76% accuracy. At larger α (2⁻¹⁰), higher T helps slightly (98.76% vs 98.39% for T=8 vs T=1), suggesting extra inference steps can compensate for less-tuned learning rates. The compute-normalized view (x-axis = step × T) shows T=1 is the most efficient use of compute.

### Gamma sweep (inference learning rate)

Clear optimum at γ=0.5 across all α values. The interaction between γ and α is important: at γ=1.0, only the smallest α (2⁻¹⁴) achieves reasonable accuracy (97.8%). This suggests the product α·γ is the effective coupling strength, and too-aggressive values in both cause instability.

### EMA beta sweep

At streaming-tuned α values (2⁻¹⁴ to 2⁻¹⁰), β ≥ 0.5 all achieve ~98.76%. β=0.0 (forward-init) drops to 84-86% — but this is because forward-init requires much larger α (2⁻⁶), not because forward-init is inherently worse. The sweep demonstrates that you cannot decouple the variant choice from the learning rate.

### Rotation speed (δ sweep)

Streaming iPC accuracy decreases gracefully with rotation speed: 99.64% (δ=1°) → 98.76% (δ=5°) → 96.26% (δ=15°). Forward-init shows the same trend but at lower accuracy: 99.48% → 97.25% → 93.09%. At all rotation speeds, the streaming variant outperforms forward-init, with the gap increasing at faster rotation — the opposite of the initial prediction that forward-init should recover its advantage when temporal correlation weakens.


## Limitations of the Benchmark

### The task overwhelmingly rewards stability over plasticity

Out of every 72 evaluation steps (at δ=5°), 71 test "did you keep your prediction the same?" and only 1 tests "did you adapt to new information?" The metric is ~98.6% dominated by the stability component. Methods that actively learn and update every step (BP) get penalized for the prediction instability that learning introduces. Methods with persistent state (streaming iPC) get rewarded for inertia.

### The output layer can solve the task alone

When the label is constant for dozens of steps, the output layer can learn a mapping from (stale but temporally correlated) hidden representations to the correct label without needing the hidden layers to track the input accurately. The hidden layers may drift passively while the output layer does all the work. This is especially problematic because we measure classification accuracy, not representational quality — and the energy metric, while more informative, is not the primary evaluation criterion.

### Information propagation is untested

A key concern with streaming iPC is that with T=1, new input information at layer L propagates only one layer per step. In a 5-layer network, it takes ~4 steps for input information to reach layer 1. With a constant label, this latency is irrelevant — the output layer doesn't need updated hidden representations because the answer isn't changing. A task where the target changes every step would directly test whether the hidden layers contribute meaningful, input-dependent computation.

### Streaming iPC's advantage may be inertia, not learning

The persistent value nodes could be functioning as a simple smoothing mechanism: by not changing much between steps, they avoid the noise that forward-init and BP introduce through recomputation and weight updates. In a regime where not changing is almost always correct (98.6% of the time), this smoothing *is* the optimal strategy. The question is whether the same mechanism would help when the target changes frequently and the network actually needs to adapt.


## Motivation for the Next Task

The Rotating MNIST results show that streaming iPC works without breaking, but they don't convincingly show that it works *because of* the persistent value nodes' representational content. To test the streaming hypothesis properly, we need a task where:

1. **The prediction target changes every step** — so "be sticky" is not a viable strategy
2. **Temporal correlation in inputs is genuinely useful** — so persistent value nodes carry meaningful information
3. **The trivial baseline is weak** — so the interesting variation isn't buried in noise
4. **The "label" is not immediately available** — so the network can't just learn a simple reactive mapping from the clamped target

A TD value prediction task on a random walk satisfies all four criteria. The value V(s) changes at every state transition, temporal correlation is natural (you move one step at a time), no trivial baseline exists, and the target is a bootstrap estimate that depends on the network's own predictions. This is also closer to the biological motivation for predictive coding — brains don't receive ground-truth labels, they make predictions about future outcomes and update based on prediction errors.

See [td_prediction/README.md](../../td_prediction/README.md) for the experimental design.
