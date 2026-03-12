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

## The Signed Conservation Gap

Approach B only achieves absolute conservation ($\sum \lvert U_{k \leftarrow j} \rvert = \lvert U_j \rvert$). The signed sum can shrink at each layer through cancellation. Approaches C and E achieve signed conservation ($\sum U_{k \leftarrow j} = U_j$), making utility comparable across layers.

---

## Comparison

| | A | B | C | D | **E** |
|---|---|---|---|---|---|
| **Signed conservation** | Yes | No (absolute only) | Yes | No | **Yes** |
| **Correct signs (harmful case)** | No | Yes | Yes (when well-conditioned) | No | **Yes** |
| **Bounded magnitudes** | Yes | Yes | No (can amplify) | Yes | **Yes** |
| **Same formula every layer** | No | Yes | Yes (with fallback) | No | **Yes (with fallback)** |
| **Activation derivative needed** | No | Yes | Yes | Yes | **No** |

Approach E resolves the three-way tradeoff between signed conservation, correct child signs, and bounded magnitudes that forces A/B/C to sacrifice one property.

---

## Files

- `initial_notes.md` -- early brainstorming and problem framing
- `multilayer_signed_utility.md` -- detailed derivations, worked examples, and formal algorithm specification
- `multilayer_experiment.py` -- experiment comparing approaches A, B, C, E, UPGD, and SI on nonlinear tracking task
