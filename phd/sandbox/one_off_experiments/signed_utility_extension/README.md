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

Implemented in `SignedCBPTracker` in `phd/feature_search/core/feature_recycling.py` (line 737) and demonstrated in `signed_utility_experiment.py`.

## Multi-Layer Extension: Shared Output-Layer Step

All three approaches below share the same first step. At the output layer, we work in pre-activation space:

$$z_{\text{out}}^* = f^{-1}(y^*)$$

$$e = z_{\text{out}}^* - z_{\text{out}}$$

$$c_j = w_{j,\text{out}} \cdot a_j$$

$$U_j = \lvert e + c_j \rvert - \lvert e \rvert$$

This is the standard signed utility formula, just applied to pre-activations. The property $\sum_j U_j \leq \lvert z_{\text{out}}^* \rvert$ holds as usual.

The question is: **how to propagate these utilities backward through hidden layers.**

---

## Approach A: Proportional Redistribution

### Description

The simplest option. Each unit distributes its utility to its children in proportion to the magnitude of their contributions, preserving the parent's sign:

For each unit $j$ with utility $U_j$, for each child $k$:

$$U_{k \leftarrow j} = U_j \cdot \frac{\lvert c_{k \to j} \rvert}{\sum_m \lvert c_{m \to j} \rvert}$$

where $c_{k \to j} = w_{kj} \cdot a_k$.

A unit's total utility is the sum from all parents: $U_k = \sum_j U_{k \leftarrow j}$.

### Pros
- **Exactly conserves utility**: $\sum_k U_{k \leftarrow j} = U_j$ with no normalization trick needed
- **Simple to implement and reason about**: one line of math
- **O(1) per connection**
- **No activation function derivative needed**: works identically for any activation

### Cons
- **All children inherit the parent's sign.** If a parent has negative utility (is harmful), every child gets negative utility, regardless of whether the child is causing or mitigating the harm. This is the fundamental limitation.
- **Different formula than the output layer.** The output layer uses the signed utility formula; redistribution uses proportional splitting. Not a "uniform" approach.

### When It Gives Wrong Answers

When a parent is harmful and its children contribute in *different directions*, Approach A cannot tell which child is causing vs. fighting the harm. A child that partially mitigates a harmful parent still receives negative utility.

> Worked examples: [multilayer_signed_utility.md, Section 4](multilayer_signed_utility.md#4-approach-a-proportional-redistribution)

---

## Approach B: Recursive Signed Utility (Recommended)

### Description

Apply the signed utility formula at every layer by defining a "pseudo-error" for each hidden unit based on its parent-level utility:

For each unit $j$ with utility $U_j$:

$$e_j = \frac{\lvert U_j \rvert}{f'(z_j)}$$

For each child $k$:

$$c_{k \to j} = w_{kj} \cdot a_k$$

$$s_{k \to j} = \lvert e_j + c_{k \to j} \rvert - \lvert e_j \rvert$$

$$S = \sum_k \lvert s_{k \to j} \rvert$$

$$U_{k \leftarrow j} = s_{k \to j} \cdot \frac{\lvert U_j \rvert}{S}$$

The pseudo-error $e_j$ is always positive. It represents "how much should this unit's pre-activation increase to improve the network's output?" The magnitude comes from the parent-level utility, and the $1 / f'(z_j)$ factor converts from activation-space to pre-activation-space.

The normalization step ($\cdot \lvert U_j \rvert / S$) ensures $\sum_k \lvert U_{k \leftarrow j} \rvert = \lvert U_j \rvert$ -- utility is conserved in absolute value.

### Why $e_j$ Is Always Positive

This was the key design decision, arrived at after testing several sign conventions. When a unit has *negative* utility (it's harmful), it means its activation should move in the direction that reduces harm. For a unit producing too-negative an activation (which hurts the output), $e_j > 0$ means "the pre-activation should increase." The signed utility formula then naturally gives:
- Positive scores to children that push the pre-activation up (toward improvement)
- Negative scores to children that push it down (causing the harm)

If we instead used $e_j = U_j / f'(z_j)$ (preserving sign), the pseudo-error for harmful units would point *further into the harmful direction*, producing inverted sign assignments.

> Detailed derivation of why the sign flip matters, with a worked counterexample: [multilayer_signed_utility.md, Sections 5.3-5.6](multilayer_signed_utility.md#5-approach-b-recursive-signed-utility) (the "Clean Algorithm" and "Harmful Case (Revised, no sign flip)" subsections)

### Pros
- **Same formula at every layer**: signed utility formula + normalization
- **Correctly identifies which children cause vs. mitigate harm**: unlike A, a child that partially counteracts a harmful parent gets positive utility
- **Conserves utility** (via normalization): $\sum_k \lvert U_{k \leftarrow j} \rvert = \lvert U_j \rvert$
- **O(1) per connection**
- **Graceful saturation behavior**: when $f'(z_j) \to 0$ (saturated activation), $e_j \to \infty$, and the formula converges to a signed version of Approach A (proportional to $\lvert c_{k \to j} \rvert$ with signs based on $\text{sign}(c_{k \to j})$). This is a smooth degradation, not a failure mode.

### Cons
- **Requires $f'(z_j)$ and $f^{-1}(y^*)$**: needs the activation function's derivative and the inverse (for the output layer). Standard activations (sigmoid, tanh, ReLU) all have these, but it's an assumption.
- **Normalization is an approximation**: the raw signed utility sum $\sum_k s_{k \to j}$ doesn't naturally equal $\lvert U_j \rvert$, so we force conservation via normalization. This means the per-child values are *scaled* versions of the true counterfactual utilities.
- **$e_j$ direction is a heuristic**: "always positive" works correctly in the examples tested, but the pseudo-error doesn't have the clean interpretation of the output-layer error (where there's a real target). It could produce unexpected behavior in networks with unusual structure.

### Edge Cases
- **$f'(z_j) = 0$** (full saturation): $e_j \to \infty$, formula degrades to signed Approach A. Well-behaved.
- **$U_j = 0$**: $e_j = 0$, all $s_{k \to j} = \lvert c_{k \to j} \rvert$, normalization distributes $0$. All children get $U_{k \leftarrow j} = 0$. Well-behaved.
- **$S = 0$** (all contributions zero): set all $U_{k \leftarrow j} = 0$.
- **$f^{-1}(y^*)$ undefined** (e.g., $y^* = 0$ or $1$ for sigmoid): clip to a large finite value.

### Budget Property

At each redistribution, $\sum_k \lvert U_{k \leftarrow j} \rvert = \lvert U_j \rvert$. So utility magnitude is conserved at every layer. The maximum total utility across the whole network is bounded by $\text{depth} \cdot \lvert z_{\text{out}}^* \rvert$, where depth is the longest path from output to input. This makes utility values comparable across layers.

> Full worked examples (linear and sigmoid): [multilayer_signed_utility.md, Sections 5 and 8](multilayer_signed_utility.md#5-approach-b-recursive-signed-utility)
>
> Formal algorithm specification: [multilayer_signed_utility.md, Section 9](multilayer_signed_utility.md#9-final-algorithm-approach-b-recursive-signed-utility)

---

## Approach D: Gradient-Weighted Utility

### Description

Use each child's gradient-based influence on the parent's activation to distribute utility:

For each unit $j$ with utility $U_j$, for each child $k$:

$$g_{k \to j} = f'(z_j) \cdot w_{kj} \cdot a_k$$

$$U_{k \leftarrow j} = U_j \cdot \frac{g_{k \to j}}{\sum_m \lvert g_{m \to j} \rvert}$$

This is a first-order Taylor approximation: "if child $k$'s activation changed slightly, how much would parent $j$'s activation change?"

### Pros
- **Preserves sign**: children with negative gradient-contributions get opposite-sign utility
- **Conserves utility** (via normalization)
- **O(1) per connection**
- **Naturally incorporates the activation derivative**: saturated parents automatically weight children less

### Cons
- **Gradients measure marginal sensitivity, not actual contribution.** This is the critical flaw. A child can have a large *current contribution* to a parent (it's the reason the parent has its current value) while having a gradient that points in the *opposite direction*. The gradient tells you what a small perturbation would do; it doesn't tell you what the child is actually doing.
- **Gives wrong answers in the harmful-feature case.** In the tested example, a child that is the primary *cause* of a parent's harmful activation gets *positive* utility, because its gradient opposes the parent's current value. The gradient correctly says "increasing this child would decrease the parent" but ignores that the child at its *current value* is what made the parent harmful.

### Why It Fails

The root cause is a distinction between **marginal** and **actual** contribution. Signed utility (Approach B) measures actual contribution: "what would the error be if this child's contribution were removed?" Gradient weighting measures marginal contribution: "what would happen if this child's value changed by epsilon?" These diverge when a child's contribution has a different sign than its gradient (e.g., a child contributes $c = -2.0$ but its gradient is also negative, meaning increasing the child would push even more negative -- the gradient and the contribution both push the parent down, but the *utility formula* based on the gradient assigns opposite credit from what the actual-contribution formula assigns).

A variant using actual contributions instead of gradients ($g_{k \to j} = c_{k \to j}$) reduces to a signed version of Approach A, which has the same double-negative problem.

> Detailed failure analysis with worked example: [multilayer_signed_utility.md, Section 6](multilayer_signed_utility.md#6-approach-d-gradient-weighted-utility)

---

## Summary

| | Approach A | Approach B | Approach C | Approach D |
|---|---|---|---|---|
| **Formula** | Proportional split | Signed utility + abs normalization | Signed utility + signed normalization | Gradient-weighted split |
| **Signed conservation** | Yes | No (absolute only) | **Yes** | No |
| **Correct signs (harmful case)** | No | Yes | **Yes (when well-conditioned)** | No |
| **Bounded magnitudes** | Yes | Yes | No (can amplify) | Yes |
| **Same formula every layer** | No | Yes | Yes (with fallback) | No |
| **Activation derivative needed** | No | Yes | Yes | Yes |
| **O(1) per connection** | Yes | Yes | Yes | Yes |
| **Saturation behavior** | N/A | Degrades to signed-A | Degrades to A | Gradients vanish |
| **Recommendation** | Baseline | Correct signs baseline | **Primary candidate** | Not recommended |

## The Signed Conservation Gap

The original desiderata (Section 1) include **finite budget** with cross-layer comparison. The strongest version of this property is **signed conservation**: if the network reduces absolute error by $E = \lvert y^* \rvert - \lvert y^* - \hat{y} \rvert$, then the utilities at *every layer* should sum to exactly $E$. This means utility is a complete decomposition of credit — nothing is created or lost as we propagate backward.

Approach B only achieves **absolute conservation**: $\sum_k \lvert U_{k \leftarrow j} \rvert = \lvert U_j \rvert$. The *signed* sum $\sum_k U_{k \leftarrow j}$ can differ from $U_j$ because the normalization $s_k \cdot \lvert U_j \rvert / \sum \lvert s_k \rvert$ distributes $\lvert U_j \rvert$ worth of magnitude but allows the signed total to shrink when children have mixed-sign contributions. In practice, this means $\sum U_{\text{input}} < \sum U_{\text{hidden}} < E_{\text{reduced}}$ — utility "leaks" at each layer through cancellation.

Additionally, even at the output layer, the leave-one-out utilities $U_j = \lvert e + c_j \rvert - \lvert e \rvert$ don't sum to $E = \lvert y^* \rvert - \lvert e \rvert$. The leave-one-out sum is $\sum_j (\lvert e + c_j \rvert - \lvert e \rvert)$, which equals error reduced only when all features contribute in the same direction (no overshooting). So signed conservation requires fixing *both* the output layer and the redistribution.

### Why signed conservation matters

Without it, a unit's utility depends on which layer it sits in. A hidden unit with utility 0.5 and an input with utility 0.5 don't represent the same amount of credit for reducing error — the hidden unit's utility was attenuated when redistributed to inputs. This breaks cross-layer comparison, which was the whole point of having a budget.

---

## Approach C: Signed-Conserving Redistribution

### The key formula

Every proportional redistribution that preserves the signed sum must take the form:

$$U_{k \leftarrow j} = s_k \cdot \frac{U_j}{\sum_m s_{m \to j}}$$

This is the *only* scaling that gives $\sum_k U_{k \leftarrow j} = U_j$ while keeping utilities proportional to the raw scores $s_k$. (Proof: $\sum_k s_k \cdot U_j / \sum s_m = U_j$.)

### Full algorithm

**Output layer** (rescaled to sum to error reduced):

$$e = y^* - \hat{y}$$

$$E = \lvert y^* \rvert - \lvert e \rvert$$

$$c_j = w_{\text{out},j} \cdot a_j$$

$$u_j = \lvert e + c_j \rvert - \lvert e \rvert$$

$$U_j = u_j \cdot \frac{E}{\sum_m u_m}$$

**Hidden layers** (signed normalization):

$$e_j = \frac{\lvert U_j \rvert}{f'(z_j)}$$

$$s_k = \lvert e_j + c_{k \to j} \rvert - \lvert e_j \rvert$$

$$S_{\text{signed}} = \sum_m s_{m \to j}$$

$$U_{k \leftarrow j} = s_k \cdot \frac{U_j}{S_{\text{signed}}}$$

### The cancellation problem

When children's contributions nearly cancel ($\sum s_k \approx 0$), the scaling factor $U_j / \sum s_k$ blows up. This happens when the unit's pre-activation $z_j \approx 0$ (because for large pseudo-error, $s_k \approx c_k$ and $\sum c_k = z_j$). In this regime, the unit is at the steepest part of the sigmoid, and small differences between children get amplified into huge utility swings.

This is a fundamental tension, not a fixable edge case:
- **Signed conservation** requires $\sum U_{k \leftarrow j} = U_j$
- **Correct child signs** means some $U_{k \leftarrow j} > 0$ and some $< 0$
- **Bounded magnitudes** requires individual $\lvert U_{k \leftarrow j} \rvert$ stays reasonable

When children partially cancel, satisfying the first two forces the third to fail. You cannot decompose a large signed total into mixed-sign pieces that sum to it without some pieces being larger than the total.

### Practical solution: fallback to Approach A

Approach A *also* has signed conservation ($\sum_k U_j \cdot \lvert c_k \rvert / \sum \lvert c_m \rvert = U_j$), just without child-level sign discrimination. Use it as a fallback when the signed normalization is ill-conditioned:

$$r = \frac{\lvert S_{\text{signed}} \rvert}{\sum \lvert s_k \rvert}$$

If $r > \text{threshold}$ (children mostly aligned):

$$U_{k \leftarrow j} = s_k \cdot \frac{U_j}{S_{\text{signed}}}$$

Otherwise (children nearly cancel):

$$U_{k \leftarrow j} = U_j \cdot \frac{\lvert c_k \rvert}{\sum \lvert c_m \rvert}$$

Both branches preserve $\sum U_{k \leftarrow j} = U_j$. The threshold controls the tradeoff: lower threshold allows more sign discrimination but risks larger magnitudes.

### Properties

| | Approach A | Approach B | **Approach C** |
|---|---|---|---|
| **Signed conservation** | Yes | No (absolute only) | **Yes** |
| **Correct child signs** | No | Yes | **Yes (when well-conditioned)** |
| **Bounded magnitudes** | Yes | Yes | No (can amplify near cancellation) |
| **Same formula every layer** | No | Yes | Yes (with fallback) |
| **O(1) per connection** | Yes | Yes | Yes |

### Why not sequential decomposition?

An alternative: remove children one at a time in some order and assign utility by telescoping ($U_k = \lvert R_{\text{prev}} \rvert - \lvert R_{\text{prev}} - c_k \rvert$). This sums to the right total by construction. But the per-child values depend on the ordering — the first child removed gets credit for the "easy" part of the error, and the last gets credit for the "hard" part. This makes the decomposition arbitrary.

---

## Recommended Plan

1. **Implement Approach C** as the primary algorithm (signed conservation with A-fallback)
2. **Keep Approaches A and B** as baselines for comparison
3. Test on the tracking task from `multilayer_experiment.py`
4. Verify: Figure 3 (budget plot) should show $\sum U_{\text{hidden}} \approx \sum U_{\text{input}} \approx E_{\text{reduced}}$ for Approach C
5. Compare sign discrimination across approaches when teacher weights drift

## Files

- `initial_notes.md` -- early brainstorming and problem framing
- `multilayer_signed_utility.md` -- detailed derivations, worked examples, and formal algorithm specification
- `multilayer_experiment.py` -- experiment comparing approaches A, B, C on nonlinear tracking task
