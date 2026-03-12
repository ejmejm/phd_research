# Multi-Layer Signed Utility: Where We Are

## The goal

We want a per-unit utility measure for pruning in multi-layer networks. The measure must:

1. **Be signed** — distinguish features that help from features that actively hurt
2. **Conserve across layers** — utilities at every layer sum to the same total (the error reduced), so we can directly compare a hidden unit's utility to an input's utility
3. **Scale with error reduced** — if at one step the reducible error is 10 and at another it's 1, features contributing to the first should be credited 10x more
4. **Be cheap** — O(1) per connection, no extra forward passes

Conservation is critical for two reasons beyond cross-layer comparison. First, it **forces the distinction between useless and harmful**. If the error reduced is $E$ and relevant features account for utility $> E$, then irrelevant features *must* have negative total utility — the budget constraint demands it. Without conservation, the missing budget just vanishes and irrelevant features sit at zero. Second, the budget scales with $E$ automatically, so steps with large reducible error naturally weight more in the running average. This is property 3 for free.

## The experiment

A 20-input student (→ 16 hidden sigmoid → 1 linear output) learns online from a 5-input teacher with LTU activations and binary weights. Only inputs 0-4 carry signal; 5-19 are noise. Teacher input weight signs flip every 50 steps, creating nonstationarity. 100k steps, tested with SGD and Autostep.

## What the four approaches do

All approaches share the output-layer step: $U_j = |e + c_j| - |e|$ for each hidden unit, where $e$ is prediction error and $c_j = w_j \cdot a_j$.

**Approach A (Proportional):** Distributes each parent's utility to children by $|c_k| / \sum |c_m|$. Conserves the signed sum. But all children inherit the parent's sign — a child fighting a harmful parent still looks harmful.

**Approach B (Recursive signed):** Applies the signed utility formula recursively using a pseudo-error $e_j = |U_j|/f'(z_j)$, giving raw scores $s_k = |e_j + c_k| - |e_j|$. Normalizes by $\sum |s_k|$ (absolute sum), so $\sum |U_{k \leftarrow j}| = |U_j|$. Correctly discriminates which children cause vs. fight harm at the per-parent level. But the signed sum $\sum U_{k \leftarrow j} \neq U_j$ — utility leaks through cancellation at every layer.

**Approach C (Signed-conserving):** Same raw scores as B, but normalizes by $\sum s_k$ (signed sum), so $\sum U_{k \leftarrow j} = U_j$ exactly. Output-layer utilities are rescaled so $\sum U_j = E_{\text{reduced}}$. Falls back to Approach A when children nearly cancel.

**Approach D (Gradient-weighted):** Distributes by gradient influence. Fails because gradients measure marginal sensitivity, not actual contribution. Not tested.

## Results: what works

**All methods correctly rank relevant above irrelevant.** Relevant/irrelevant utility ratios range from 4x (Contribution) to 70x (Approach B), depending on method and seed. The signal is robust.

**Approach B produces the cleanest, most stable traces.** Relevant inputs (blue) are consistently positive, well-separated from irrelevant inputs (red, near zero). Traces stabilize within ~10k steps with no spikes. Typical ratio 20-70x.

**Approach C achieves signed conservation at the hidden layer.** The budget plot confirms $\sum U_{\text{hidden}} \approx E_{\text{reduced}}$ (by construction of the output rescaling). Input-layer conservation is also much better than B, though not perfect due to the A-fallback.

## Results: what doesn't work

### Approach B cannot show irrelevant features as negative

In every run, B's irrelevant input traces sit at approximately zero — slightly positive, never clearly negative. This is a fundamental limitation of the absolute normalization.

The formula $U_{k \leftarrow j} = s_k \cdot |U_j| / \sum |s_m|$ uses $|U_j|$ — always positive — so each child's utility sign depends only on its own $s_k$, not on whether the parent is helpful or harmful. For irrelevant inputs, $s_k$ is small and random, so $U_k \approx 0$ after summing over parents.

But worse: without conservation, there's nothing to *force* irrelevant inputs to be negative. In a conserving method, if relevant inputs account for more than $E_{\text{reduced}}$, irrelevant inputs must make up the deficit with negative utility. B has no such constraint — the budget just leaks and no one gets blamed.

This means B cannot distinguish between:
- An input with a small weight because it's unimportant but slightly useful
- An input whose noise contribution is actively increasing prediction error

For pruning, this distinction is exactly what we need.

### Approach C has transient spikes

C's utility traces show periods where individual inputs spike from ~0.1 to -2 or +4. These happen during teacher sign flips and affect both relevant and irrelevant inputs.

There are two contributing causes, and they are importantly different:

**1. Genuine signal volatility (correct behavior).** When teacher signs flip, the network's predictions are suddenly wrong. Error spikes. $E_{\text{reduced}} = |y^*| - |e|$ goes negative — the network is overshooting. Features that contributed to the now-wrong prediction genuinely deserve negative utility. This is NOT a bug. A feature that was helpful before the flip is now harmful, and its utility should reflect that. The "spike to -2" may be telling the truth: during this transient, this feature really is actively hurting.

**2. Numerical amplification near cancellation (a real problem).** The signed normalization $U_j / \sum s_k$ blows up when children's raw scores nearly cancel ($\sum s_k \approx 0$). This happens when $z_j \approx 0$ (the most active part of the sigmoid). The alignment threshold prevents the worst cases, but at the boundary (alignment ≈ 0.1), the amplification factor is ~10x per hidden unit. These amplifications sum across hidden units, producing input-level utilities far larger in magnitude than the error reduced. This violates the spirit of the budget even when the sum is correct.

The key question: **how much of C's volatility is signal vs. noise?** If most of the spikes are genuine (features really are transiently harmful during sign flips), then C is giving us important information that B suppresses. If most are numerical artifacts from near-cancellation, then C is corrupting the signal.

## The fundamental tradeoff

There is a mathematical impossibility when children partially cancel. If hidden unit $j$ has utility $U_j$ and its children have mixed-sign contributions that nearly cancel:

- **Signed conservation** requires $\sum_k U_{k \leftarrow j} = U_j$
- **Correct child signs** means some $U_{k \leftarrow j} > 0$ and some $< 0$
- **Bounded individual magnitudes** requires $|U_{k \leftarrow j}| \leq M$ for some reasonable $M$

When the positive and negative children nearly cancel, you're decomposing a number into mixed-sign pieces that must sum to it. Some pieces must be larger than the total. This is algebra, not a fixable bug.

| | Conservation | Correct signs | Bounded magnitudes |
|---|---|---|---|
| Approach A | Yes | No | Yes |
| Approach B | No | Yes | Yes |
| Approach C | Yes | Yes | No |
| **Approach E** | **Yes** | **Yes** | **Yes** |

Approaches A–C each sacrifice exactly one property. No **normalization-based** redistribution of the form $U_{k \leftarrow j} = \alpha_k \cdot U_j / \sum \alpha_m$ can achieve all three when children cancel. Approach E sidesteps this by eliminating normalization entirely — it chooses the pseudo-error $e_j$ so that the raw scores already sum to $U_j$, requiring no division.

## Analysis: what does pruning actually need?

For pruning, we need to:

1. **Identify the least useful units at each layer** — within-layer ranking
2. **Compare across layers** — should we prune an input or a hidden unit?
3. **Distinguish harmful from useless** — actively harmful features should be pruned first
4. **Decide quickly** — the faster utility stabilizes, the sooner we can prune

Requirements 1-2 need conservation (or something close to it). Requirement 3 needs sign discrimination. Requirement 4 needs stable traces.

Approach B is strong on 1 and 4 (clean stable traces, good within-layer ranking) but fails on 2 (no conservation → can't compare across layers) and partially on 3 (irrelevant features near zero, not clearly negative).

Approach C is strong on 1-3 (conservation, sign discrimination, correct cross-layer scaling) but the amplification spikes threaten 4 (slow to stabilize because transient spikes corrupt the running average).

## Open directions

### Is Approach C's volatility mostly signal?

The "spikes" in C deserve closer examination. Some of the volatility is genuine: during sign flips, the utility landscape really does change dramatically, and features that were helpful become harmful. If we separate the genuine volatility from the numerical amplification, the residual might be acceptable.

One diagnostic: track the alignment ratio over time. If the spikes mostly occur when alignment is above threshold (genuine high-cancellation regime) vs. when alignment is near the threshold (borderline numerical issue), that tells us the split.

Another: compare C's input utility traces on a stationary task (no sign drifts). If the spikes disappear, they're driven by the nonstationarity, not numerical issues.

### Can we damp the amplification without breaking conservation?

The amplification comes from dividing by $\sum s_k$ when it's small. Any modification to the denominator breaks conservation. But there might be other approaches:

**Soft blending between A and C.** Instead of a hard threshold, weight the A and C redistributions by a smooth function of alignment:

$$U_{k \leftarrow j} = \alpha(r) \cdot U_{k \leftarrow j}^C + (1 - \alpha(r)) \cdot U_{k \leftarrow j}^A$$

Both branches conserve, so the blend conserves. By making $\alpha(r)$ smooth (e.g., sigmoid of alignment ratio), we avoid the hard cutoff that creates worst-case amplification at the boundary. As alignment decreases, we smoothly transition to A's stable redistribution.

This still doesn't eliminate amplification for large alignment ratios where $\sum s_k$ is genuinely small. But it removes the discontinuity at the threshold.

**Observation:** Approach A also conserves signed utility. The problem with A is that it doesn't discriminate child signs. But it IS correct about the total magnitude and the parent's sign. So falling back to A when signed normalization is ill-conditioned is not "giving up" — it's acknowledging that when children nearly cancel, the detailed per-child sign assignment is inherently uncertain, and the best we can confidently say is the parent-level attribution.

### Can we fix Approach B with post-hoc rescaling?

After computing B's utilities (correct signs, bounded, stable), rescale each layer's utilities to sum to $E_{\text{reduced}}$:

$$U_k^{\text{corrected}} = U_k \cdot \frac{E_{\text{reduced}}}{\sum_m U_m}$$

This restores the cross-layer budget and cross-time scaling. It preserves within-layer ranking. But it does NOT fix the harmful-feature problem: irrelevant inputs near zero stay near zero after rescaling — the correction is a uniform scale factor that doesn't change signs or relative magnitudes.

So this addresses conservation (requirements 2-3 above) but not sign discrimination (requirement 3). Not a complete solution.

### Does Approach B's sign propagation actually fail here?

B uses $|U_j|$ in the normalization, which means the parent's sign is lost. But B's pseudo-error construction is designed to handle this: when $U_j < 0$ (harmful parent), $e_j = |U_j|/f'(z_j) > 0$, and the signed scores $s_k$ are supposed to correctly identify which children *cause* the harm (negative $s_k$) vs. *fight* it (positive $s_k$).

So at the per-parent level, B does produce negative utilities for children that cause harm. The problem is that when we sum over parents ($U_k = \sum_j U_{k \leftarrow j}$), an irrelevant input gets small random contributions from each parent that approximately cancel to zero.

This isn't a failure of B's sign logic — it's that the *instantaneous* effect of an irrelevant input really is approximately zero on any given step (it helps some parents and hurts others by similar amounts). The fact that irrelevant inputs are harmful *on average* (by Jensen's inequality: $E[|e + c|] \geq |e|$ when $c$ is zero-mean noise) is a second-order effect that's too small to measure step by step.

Conservation amplifies this second-order effect into a clear signal by forcing the budget to balance. Without conservation, the effect is lost in noise.

### Is there a fundamentally different decomposition?

All approaches A-C decompose utility by distributing each parent's utility to its children. This creates the cancellation problem when children have mixed signs.

What about a non-decomposition approach? Define utility directly at each layer without propagating from the layer above:

- **Input-level utility**: for each input $k$, compute $|e_{\text{without input } k}| - |e|$ using a forward pass with input $k$ zeroed out. This is the true leave-one-out utility at the input level and naturally sums to at most $|y^*|$.

This is expensive (one extra forward pass per input per step), but it gives a clean ground truth to compare against. If this matches C on average but without spikes, then C's amplification is pure noise. If they differ, the decomposition approach has a systematic bias.

A cheaper approximation: use the chain rule. The effect of zeroing input $k$ on the output is approximately $\sum_j w_{\text{out},j} \cdot f'(z_j) \cdot W_{jk} \cdot x_k$. This is first-order (gradient-based), which we've seen has problems, but it could serve as a diagnostic.

### Summary of the state of things

- **Approach B works well for within-layer ranking** but cannot compare across layers and cannot identify actively harmful features
- **Approach C has the right algebraic properties** for cross-layer comparison and harmful feature identification, but numerical amplification near cancellation produces spikes that corrupt the running average
- **The amplification is structurally unavoidable** for any normalization-based redistribution — the three-way tradeoff is real for approaches that fix $e_j$ and then divide by $\sum s_k$
- **Approach E (calibrated pseudo-error)** sidesteps the tradeoff entirely by choosing $e_j$ so that the raw scores already sum to the target. Achieves conservation, correct signs, and bounded magnitudes simultaneously. This is the most promising candidate and should be implemented and tested next

### Approach E: Calibrated Pseudo-Error (new direction)

The three-way tradeoff above applies to **normalization-based** approaches: those that compute raw scores with a fixed pseudo-error and then divide by their sum to enforce conservation. The division is the source of the blowup.

Approach E takes a different route. The function $g(e) = \sum_k (|e + c_k| - |e|)$ varies continuously from $\sum |c_k|$ (at $e = 0$) to $\sum c_k = z_j$ (as $e \to \infty$). Instead of fixing $e_j$ and normalizing, find $e_j$ such that $g(e_j) = U_j$ — the raw scores sum to the target utility by construction. No normalization, no division by small numbers, no blowup.

This achieves all three properties simultaneously:
- **Conservation**: $\sum s_k = U_j$ by construction
- **Correct signs**: same signed utility formula as B and C
- **Bounded magnitudes**: each $|s_k| \leq |c_k|$, no amplification

The calibrated $e_j$ also produces a natural smooth transition:
- When the target is close to $\sum |c_k|$, $e_j$ is small, and scores $\approx |c_k|$ (A-like: all positive, proportional)
- When the target is close to $z_j$, $e_j$ is large, and scores $\approx \text{sign}(e_j) \cdot c_k$ (B-like: full sign discrimination)

The transition is data-driven — no thresholds, no blending weights. When children nearly cancel ($z_j \approx 0$), the feasible range for $U_j$ is narrow, and $e_j$ stays small, automatically falling back to an A-like decomposition. This is the correct behavior: when children cancel, per-child sign attribution is inherently uncertain.

The cost is ~15–20× more computation per step (bisection to find $e_j$), but this is fully vectorizable and the constant is small. A fallback to A is needed when $U_j$ falls outside the feasible range $[-|z_j|, \sum |c_k|]$, but with calibrated output-layer utilities this should be rare.

The key experiment: implement Approach E and compare its traces against C and true LOO. It should match C's conservation and LOO's sign discrimination while being as stable as B.

### True LOO comparison (completed)

Comparing against true leave-one-out utilities (via 20 extra forward passes per step, EMA-traced) confirmed:
- True LOO shows irrelevant features as **slightly negative** (Jensen's inequality effect), confirming that conservation should force this.
- True LOO traces are **noisy but spike-free** — C's spikes are numerical amplification artifacts, not genuine signal.
- True LOO budget roughly tracks error reduced, though not perfectly (LOO is not an exact additive decomposition due to interaction effects).
- LOO serves as the gold standard reference for evaluating decomposition methods.
