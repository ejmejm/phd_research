# Extending Signed Utility to Multi-Layer Networks

## The Problem

We have a utility measure for features in linear networks (signed utility) that tells us how much each feature contributes to reducing absolute error. We want to extend this to multi-layer networks with nonlinear activations, retaining these properties:

1. **Finite budget** -- a bounded total amount of utility, enabling cross-layer comparison
2. **Signed** -- distinguishes features that help from features that hurt
3. **Cheap** -- $O(1)$ per connection, no extra forward passes
4. **Uniform** -- ideally the same formula at every layer

## Signed Utility (Existing, Single Layer)

For a linear prediction $\hat{y} = \sum_i w_i \cdot a_i$ with target $y^*$:

$$u_i = \lvert y^* - \sum_{j \neq i} w_j \cdot a_j \rvert - \lvert y^* - \hat{y} \rvert = \lvert \text{error\_without\_i} \rvert - \lvert \text{error\_with\_all} \rvert$$

- $u_i > 0$: feature $i$ is helping (removing it would increase error)
- $u_i < 0$: feature $i$ is harmful (removing it would decrease error)
- $\sum_i u_i \leq \lvert y^* \rvert$: total utility bounded by target magnitude

Implemented in `SignedCBPTracker` in `phd/feature_search/core/feature_recycling.py` (line 737).

## Multi-Layer Extension: Shared Output-Layer Step

All approaches share the same first step. At the output layer:

$$e = z_{\text{out}}^* - z_{\text{out}}, \quad c_j = w_{j,\text{out}} \cdot a_j, \quad U_j = \lvert e + c_j \rvert - \lvert e \rvert$$

The question is: **how to propagate these utilities backward through hidden layers.**

---

## Approach A: Proportional Redistribution

Each unit distributes its utility to children in proportion to contribution magnitude, preserving the parent's sign:

$$U_{k \leftarrow j} = U_j \cdot \frac{\lvert c_{k \to j} \rvert}{\sum_m \lvert c_{m \to j} \rvert}$$

Simple and exactly conserves utility, but **all children inherit the parent's sign** — cannot distinguish which child causes vs. mitigates harm.

> Worked examples: [multilayer_signed_utility.md, Section 4](multilayer_signed_utility.md#4-approach-a-proportional-redistribution)

---

## Approach B: Recursive Signed Utility

Applies the signed utility formula at every layer using a pseudo-error derived from the parent's utility:

$$e_j = \frac{\lvert U_j \rvert}{f'(z_j)}, \quad s_{k \to j} = \lvert e_j + c_{k \to j} \rvert - \lvert e_j \rvert, \quad U_{k \leftarrow j} = s_{k \to j} \cdot \frac{\lvert U_j \rvert}{\sum_k \lvert s_{k \to j} \rvert}$$

The pseudo-error $e_j$ is always positive. For harmful units ($U_j < 0$), this means "the pre-activation should increase to reduce harm." Children pushing the pre-activation up get positive utility; those pulling it down get negative. The normalization enforces **absolute conservation**: $\sum_k \lvert U_{k \leftarrow j} \rvert = \lvert U_j \rvert$.

Correctly identifies which children cause vs. mitigate harm, unlike A. When $f'(z_j) \to 0$ (saturation), degrades smoothly to a signed version of A.

> Detailed derivation: [multilayer_signed_utility.md, Sections 5-8](multilayer_signed_utility.md#5-approach-b-recursive-signed-utility)

---

## Approach C: Signed-Conserving Redistribution

Same raw scores as B, but normalized to preserve the **signed** sum:

$$U_{k \leftarrow j} = s_k \cdot \frac{U_j}{\sum_m s_{m \to j}}$$

This is the only scaling that gives $\sum_k U_{k \leftarrow j} = U_j$ while keeping utilities proportional to $s_k$. However, when children's contributions nearly cancel ($\sum s_k \approx 0$), the division blows up. Falls back to Approach A when ill-conditioned.

---

## Approach D: Gradient-Weighted Utility (Not Recommended)

Distributes utility using gradient-based influence: $g_{k \to j} = f'(z_j) \cdot w_{kj} \cdot a_k$.

**Fails because gradients measure marginal sensitivity, not actual contribution.** A child that is the primary cause of a parent's harmful activation can get positive utility because its gradient opposes the current value. The gradient says "increasing this child would help" but ignores that the child at its current value is what caused the harm.

> Failure analysis: [multilayer_signed_utility.md, Section 6](multilayer_signed_utility.md#6-approach-d-gradient-weighted-utility)

---

## Approach E: Calibrated Pseudo-Error

Instead of computing scores and normalizing, **choose the pseudo-error $e_j$ so that raw scores already sum to the target utility**. No normalization needed.

The function $g(e) = \sum_k (\lvert e + c_k \rvert - \lvert e \rvert)$ is continuous and piecewise-linear, ranging from $\sum \lvert c_k \rvert$ at $e=0$ to $\pm z_j$ as $e \to \pm\infty$. For any feasible target $T$, there exists a unique $e_j$ such that $g(e_j) = T$, found analytically via the piecewise-linear structure (sort breakpoints, solve linear equation per segment).

This naturally interpolates between A-like behavior (small $e_j$, scores $\approx \lvert c_k \rvert$) and B-like behavior (large $e_j$, scores $\approx \text{sign}(e_j) \cdot c_k$). Falls back to A when $U_j$ is outside the feasible range (rare with calibrated output-layer utilities).

> Detailed derivation: sections above on calibration function and analytical solution

---

## Approach F: Capped Signed Redistribution

A simpler alternative to E that starts from C's signed normalization but prevents the blow-up by capping individual utilities.

**Algorithm.** For unit $j$ with utility $U_j$, distributing to children $k$:

1. Compute raw scores as in B (pseudo-error always positive):

$$e_j = \frac{\lvert U_j \rvert}{f'(z_j)}, \quad s_{k \to j} = \lvert e_j + c_{k \to j} \rvert - \lvert e_j \rvert$$

2. Apply C's signed normalization:

$$\tilde{U}_{k \leftarrow j} = s_{k \to j} \cdot \frac{U_j}{\sum_m s_{m \to j}}$$

3. Cap each child's utility magnitude at the parent's:

$$U_{k \leftarrow j} = \text{sign}(\tilde{U}_{k \leftarrow j}) \cdot \min\!\big(\lvert \tilde{U}_{k \leftarrow j} \rvert,\; \lvert U_j \rvert\big)$$

**Why this works.** When $\sum s_k$ is well-behaved (not near zero), the normalization in step 2 produces reasonable magnitudes and the cap in step 3 never activates — the method behaves identically to C with exact signed conservation. When $\sum s_k \approx 0$ (children's contributions nearly cancel), step 2 would normally produce enormous utilities; the cap clips them to $\lvert U_j \rvert$, preventing blow-up at the cost of losing exact signed conservation for that node.

**What it sacrifices.** After capping, $\sum_k U_{k \leftarrow j} \neq U_j$ in general. The signed sum will be *closer* to $U_j$ than Approach B's signed sum (which can shrink substantially through cancellation), but it won't be exact when caps activate. The total magnitude $\sum_k \lvert U_{k \leftarrow j} \rvert$ is also no longer constrained to equal $\lvert U_j \rvert$ — it can be less (utility is "lost" to capping) but never more per child.

**Edge case ($\sum s_k = 0$).** Falls back to Approach A (proportional redistribution by $\lvert c_k \rvert$).

**Compared to E:** No sorting of breakpoints, no piecewise-linear solve — just the same score computation as B/C plus a single clamp per child. $O(1)$ per connection with a much smaller constant. The tradeoff is approximate rather than exact signed conservation when the cap activates.

---

## Approach G: Target Propagation Utility

Instead of manufacturing a pseudo-error from $\lvert U_j \rvert / f'(z_j)$ (B/C/F) or solving for a calibrated one (E), compute a **real pre-activation target** by inverting the activation function — target propagation style.

**Algorithm.** For unit $j$ with utility $U_j$ and parent weight $w_{j,\text{out}}$:

1. Compute the target activation (what $a_j$ should have been to shift $j$'s output contribution by $U_j$):

$$a_j^* = a_j + \frac{U_j}{w_{j,\text{out}}}$$

2. Invert the activation to get the target pre-activation:

$$z_j^* = f^{-1}(a_j^*)$$

If $a_j^*$ is outside the range of $f$ (e.g. $a_j^* \notin (0,1)$ for sigmoid), set $z_j^* = z_j \pm 10^6$ in the direction of the needed change.

3. The pre-activation error:

$$e_j = z_j^* - z_j$$

4. Standard LOO scores with this error:

$$s_{k \to j} = \lvert e_j + c_{k \to j} \rvert - \lvert e_j \rvert$$

5. Signed normalization with cap (as in F):

$$\tilde{U}_{k \leftarrow j} = s_{k \to j} \cdot \frac{U_j}{\sum_m s_{m \to j}}, \quad U_{k \leftarrow j} = \text{clip}\!\big(\tilde{U}_{k \leftarrow j},\; -\lvert U_j \rvert,\; \lvert U_j \rvert\big)$$

**Why this is different from B/C/F.** Those approaches derive the pseudo-error from the utility magnitude and the activation derivative: $e_j = \lvert U_j \rvert / f'(z_j)$. This is a first-order linear approximation of the true nonlinear relationship. Approach G computes the exact nonlinear inverse $f^{-1}(a_j^*)$, which:

- Correctly handles the nonlinearity of the activation (large differences in pre-activation space near saturation that are small in activation space, and vice versa)
- Naturally encodes the **sign** of the needed change (the error points toward the target, not just "away from current")
- Does not need $f'(z_j)$ — it needs $f^{-1}$ instead (logit for sigmoid)

**Saturation behavior.** When the target activation is unattainable ($a_j^*$ outside the sigmoid's range), the error is set to $\pm 10^6$. In this regime, $s_k \approx \text{sign}(e_j) \cdot c_k$ — scores become proportional to signed contributions. After normalization, $U_k \approx c_k \cdot U_j / z_j$. This is a natural graceful degradation: when the activation function can't express the needed change, utility is distributed proportionally to how much each child contributes.

**Multi-parent units.** When unit $j$ feeds multiple parents $p$ with weights $w_{jp}$, the target activation shift becomes $\delta_a = \sum_p U_{j \to p} / w_{jp}$. Different parents may request conflicting shifts. For a single hidden layer (as in the experiment), each unit has one parent, so this does not arise.

---

## Approach H: Activation-Absorbed Signed Utility

Approach B always scales children's scores so that $\sum |U_{k \leftarrow j}| = |U_j|$ — forcing all of the parent's utility onto its children. But when children's contributions cancel and the activation function has $f(0) \neq 0$ (e.g., $\sigma(0) = 0.5$), the unit produces useful output even with zero net input. That utility belongs to the activation function, not the children.

**Key insight:** The activation function can act as a utility source (like a bias). Utility should be allowed to *decrease* as it propagates backward — the deficit is utility the activation function earned. Utility should never *increase* (children cannot create utility the parent doesn't have).

**Algorithm.** Same raw scores as B, with one change to the normalization:

$$e_j = \frac{\lvert U_j \rvert}{f'(z_j)}, \quad s_{k \to j} = \lvert e_j + c_{k \to j} \rvert - \lvert e_j \rvert$$

$$U_{k \leftarrow j} = s_{k \to j} \cdot \min\!\left(1,\; \frac{\lvert U_j \rvert}{\sum_m \lvert s_{m \to j} \rvert}\right)$$

The single change from B: replace $|U_j|/S$ with $\min(1, |U_j|/S)$. Don't scale up, only scale down.

- When $S \leq |U_j|$: raw scores used directly. The deficit $|U_j| - S$ is absorbed by the activation function.
- When $S > |U_j|$: scores scaled down to fit the budget. Identical to Approach B in this regime.

**Motivating example.** Sigmoid unit, $c_A = +1$, $c_B = -1$, $z = 0$, $a = 0.5$, $U_j = 2$. Pseudo-error $e_j = 8$. Scores: $s_A = +1$, $s_B = -1$. $S = 2 = |U_j|$, so $\min(1, 2/2) = 1$ — no scaling. Children get $+1$ and $-1$ (sum $= 0$). The deficit of 2 is the activation function's contribution: $\sigma(0) = 0.5$ produced the output, not the children.

**Properties:**

- **Bounded:** $\sum_k |U_{k \leftarrow j}| \leq |U_j|$. Utility never grows through propagation.
- **Correct signs:** same as B — children causing harm get negative utility, those mitigating get positive.
- **Same formula every layer** (signed utility formula + capped normalization).
- **$O(1)$ per connection.**
- **Graceful degradation at saturation:** $e_j \to \infty$, scores $\to c_k$, cap engages, reduces to signed proportional redistribution.
- **No blow-up.** When $S = 0$, all scores are zero and the min is irrelevant.

**Property relaxed (by design):** Signed conservation. $\sum_k U_{k \leftarrow j}$ may be less than $U_j$. The deficit is utility absorbed by the activation function — not lost, just not attributable to any child.

---

## Approach I: Coherence-Weighted Decomposition

Approaches C/E/F/G all wrestle with the same fundamental tension: signed conservation ($\sum U_{k \leftarrow j} = U_j$) requires dividing by $\sum s_k$, which blows up when children's contributions cancel. This tension is **intrinsic** — any decomposition proportional to scores $s_k$ must divide by their sum to conserve the signed total. No choice of pseudo-error or normalization can avoid this.

Approach I sidesteps the problem entirely by **not using the signed utility formula at hidden layers**. The LOO formula $|e + c_k| - |e|$ is meaningful at the output (where $e$ is a real error), but at hidden layers the "pseudo-error" is artificial. When the pseudo-error is large relative to contributions (the typical regime), the scores converge to $s_k \approx \text{sign}(e) \cdot c_k$ anyway — just the raw contributions. The nonlinear regime only matters when $U_j$ is small, where there's little utility to distribute.

**Core idea.** Blend two natural decompositions of $U_j$ using the **coherence** $\beta = |z_j| / \sum |c_m|$ as interpolation weight:

- **Signed decomposition** $c_k / z_j$: exact proportions, blows up when $z_j \to 0$
- **Absolute decomposition** $|c_k| / \sum|c_m|$: always safe, loses sign information

$$U_{k \leftarrow j} = U_j \left[\beta \cdot \frac{c_k}{z_j} + (1 - \beta) \cdot \frac{|c_k|}{\Sigma}\right], \quad \beta = \frac{|z_j|}{\Sigma}, \quad \Sigma = \sum_m |c_m|$$

The singularity cancels algebraically: $\beta \cdot c_k / z_j = c_k \cdot \text{sign}(z_j) / \Sigma$, giving the equivalent form:

$$U_{k \leftarrow j} = \frac{U_j}{\Sigma}\Big[\text{sign}(z_j) \cdot c_k + (1 - \beta) \cdot |c_k|\Big]$$

No division by anything that can vanish (only $\Sigma = 0$, which means all contributions are zero).

**Properties:**

- **Exact signed conservation:** $\sum_k U_{k \leftarrow j} = U_j$ (both terms in the blend sum to 1)
- **No blow-up:** the $c_k / z_j$ singularity is cancelled by the $\beta = |z_j|/\Sigma$ prefactor
- **Correct signs:** children aligned with $z_j$ get $U_j$'s sign (enabling); children opposing get the opposite sign (fighting). When $U_j < 0$, enablers get negative utility, fighters get positive
- **Bounded magnitudes:** worst-case amplification is $9/8 = 1.125\times$ per child (for 2 children); in practice negligible
- **No pseudo-error, no activation derivative:** works directly with pre-activation contributions
- **$O(1)$ per connection, simplest formula of all approaches**
- **Graceful interpolation:** $\beta = 1$ (no cancellation) → pure signed decomposition; $\beta = 0$ (total cancellation) → Approach A

---

## The Signed Conservation Gap

Approach B only achieves absolute conservation ($\sum \lvert U_{k \leftarrow j} \rvert = \lvert U_j \rvert$). The signed sum can shrink at each layer through cancellation. Approaches C and E achieve signed conservation ($\sum U_{k \leftarrow j} = U_j$), making utility comparable across layers.

---

## Comparison

| | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| **Signed conservation** | Yes | No (absolute only) | Yes | No | Yes | Approximate | Approximate | No (by design) | Yes (exact) |
| **Correct signs (harmful case)** | No | Yes | Yes (when well-conditioned) | No | Yes | Yes | Yes | Yes | Yes |
| **Bounded magnitudes** | Yes | Yes | No (can amplify) | Yes | Yes | Yes | Yes | Yes | Yes ($\leq 9/8\times$) |
| **Same formula every layer** | No | Yes | Yes (with fallback) | No | Yes (with fallback) | Yes (with fallback) | Yes (with fallback) | Yes | Yes (no fallback) |
| **Activation derivative needed** | No | Yes | Yes | Yes | No | Yes | No (uses $f^{-1}$) | Yes | No |
| **Activation absorption** | No | No | No | No | No | No | No | Yes | No |
| **Cheap (small constant)** | Yes | Yes | Yes | Yes | No (sort + solve) | Yes | Yes | Yes | Yes (cheapest) |

Approaches H and I represent two different philosophies. H acknowledges that activation functions can be utility sources (like biases) and lets utility decrease through layers — it uses B's signed utility formula but removes the forced scale-up. I achieves exact signed conservation with no blow-up by sidestepping the signed utility formula entirely. H is conceptually simpler (one-line change to B); I is algebraically simpler (no pseudo-error or derivative).

---

## Files

- `initial_notes.md` -- early brainstorming and problem framing
- `multilayer_signed_utility.md` -- detailed derivations, worked examples, and formal algorithm specification
- `multilayer_experiment.py` -- experiment comparing approaches A, B, C, E, F, G, I, UPGD, and SI on nonlinear tracking task
