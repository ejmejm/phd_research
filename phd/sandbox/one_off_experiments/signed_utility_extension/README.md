# Extending Signed Utility to Multi-Layer Networks

## The Problem

We have a utility measure for features in linear networks (signed utility) that tells us how much each feature contributes to reducing absolute error. We want to extend this to multi-layer networks with nonlinear activations, retaining these properties:

1. **Finite budget** -- a bounded total amount of utility, enabling cross-layer comparison
2. **Signed** -- distinguishes features that help from features that hurt
3. **Cheap** -- O(1) per connection, no extra forward passes
4. **Uniform** -- ideally the same formula at every layer

## Signed Utility (Existing, Single Layer)

For a linear prediction `y_hat = sum_i w_i * a_i` with target `y*`:

```
u_i = |y* - sum_{j != i} w_j * a_j| - |y* - y_hat|
    = |error_without_i| - |error_with_all|
```

- `u_i > 0`: feature `i` is helping (removing it would increase error)
- `u_i < 0`: feature `i` is harmful (removing it would decrease error)
- `sum_i u_i <= |y*|`: total utility bounded by target magnitude

Implemented in `SignedCBPTracker` in `phd/feature_search/core/feature_recycling.py` (line 737) and demonstrated in `signed_utility_experiment.py`.

## Multi-Layer Extension: Shared Output-Layer Step

All three approaches below share the same first step. At the output layer, we work in pre-activation space:

```
z_out* = f_inv(y*)               # target pre-activation
e = z_out* - z_out               # pre-activation error
c_j = w_{j,out} * a_j            # contribution of unit j

U_j = |e + c_j| - |e|            # signed utility at the output
```

This is the standard signed utility formula, just applied to pre-activations. The property `sum_j U_j <= |z_out*|` holds as usual.

The question is: **how to propagate these utilities backward through hidden layers.**

---

## Approach A: Proportional Redistribution

### Description

The simplest option. Each unit distributes its utility to its children in proportion to the magnitude of their contributions, preserving the parent's sign:

```
For each unit j with utility U_j, for each child k:
    U_{k <- j} = U_j * |c_{k->j}| / sum_m |c_{m->j}|

where c_{k->j} = w_{kj} * a_k
```

A unit's total utility is the sum from all parents: `U_k = sum_j U_{k <- j}`.

### Pros
- **Exactly conserves utility**: `sum_k U_{k <- j} = U_j` with no normalization trick needed
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

```
For each unit j with utility U_j:
    e_j = |U_j| / f'(z_j)          # pseudo-error in pre-activation space

    For each child k:
        c_{k->j} = w_{kj} * a_k
        s_{k->j} = |e_j + c_{k->j}| - |e_j|    # raw signed utility score

    S = sum_k |s_{k->j}|
    U_{k <- j} = s_{k->j} * |U_j| / S           # normalized utility
```

The pseudo-error `e_j` is always positive. It represents "how much should this unit's pre-activation increase to improve the network's output?" The magnitude comes from the parent-level utility, and the `1/f'(z_j)` factor converts from activation-space to pre-activation-space.

The normalization step (`* |U_j| / S`) ensures `sum_k |U_{k <- j}| = |U_j|` -- utility is conserved in absolute value.

### Why `e_j` Is Always Positive

This was the key design decision, arrived at after testing several sign conventions. When a unit has *negative* utility (it's harmful), it means its activation should move in the direction that reduces harm. For a unit producing too-negative an activation (which hurts the output), `e_j > 0` means "the pre-activation should increase." The signed utility formula then naturally gives:
- Positive scores to children that push the pre-activation up (toward improvement)
- Negative scores to children that push it down (causing the harm)

If we instead used `e_j = U_j / f'(z_j)` (preserving sign), the pseudo-error for harmful units would point *further into the harmful direction*, producing inverted sign assignments.

> Detailed derivation of why the sign flip matters, with a worked counterexample: [multilayer_signed_utility.md, Sections 5.3-5.6](multilayer_signed_utility.md#5-approach-b-recursive-signed-utility) (the "Clean Algorithm" and "Harmful Case (Revised, no sign flip)" subsections)

### Pros
- **Same formula at every layer**: signed utility formula + normalization
- **Correctly identifies which children cause vs. mitigate harm**: unlike A, a child that partially counteracts a harmful parent gets positive utility
- **Conserves utility** (via normalization): `sum_k |U_{k <- j}| = |U_j|`
- **O(1) per connection**
- **Graceful saturation behavior**: when `f'(z_j) -> 0` (saturated activation), `e_j -> infinity`, and the formula converges to a signed version of Approach A (proportional to `|c_{k->j}|` with signs based on `sign(c_{k->j})`). This is a smooth degradation, not a failure mode.

### Cons
- **Requires `f'(z_j)` and `f_inv(y*)`**: needs the activation function's derivative and the inverse (for the output layer). Standard activations (sigmoid, tanh, ReLU) all have these, but it's an assumption.
- **Normalization is an approximation**: the raw signed utility sum `sum_k s_{k->j}` doesn't naturally equal `|U_j|`, so we force conservation via normalization. This means the per-child values are *scaled* versions of the true counterfactual utilities.
- **`e_j` direction is a heuristic**: "always positive" works correctly in the examples tested, but the pseudo-error doesn't have the clean interpretation of the output-layer error (where there's a real target). It could produce unexpected behavior in networks with unusual structure.

### Edge Cases
- **`f'(z_j) = 0`** (full saturation): `e_j -> inf`, formula degrades to signed Approach A. Well-behaved.
- **`U_j = 0`**: `e_j = 0`, all `s_{k->j} = |c_{k->j}|`, normalization distributes `0`. All children get `U_{k <- j} = 0`. Well-behaved.
- **`S = 0`** (all contributions zero): set all `U_{k <- j} = 0`.
- **`f_inv(y*)` undefined** (e.g., `y* = 0` or `1` for sigmoid): clip to a large finite value.

### Budget Property

At each redistribution, `sum_k |U_{k <- j}| = |U_j|`. So utility magnitude is conserved at every layer. The maximum total utility across the whole network is bounded by `depth * |z_out*|`, where depth is the longest path from output to input. This makes utility values comparable across layers.

> Full worked examples (linear and sigmoid): [multilayer_signed_utility.md, Sections 5 and 8](multilayer_signed_utility.md#5-approach-b-recursive-signed-utility)
>
> Formal algorithm specification: [multilayer_signed_utility.md, Section 9](multilayer_signed_utility.md#9-final-algorithm-approach-b-recursive-signed-utility)

---

## Approach D: Gradient-Weighted Utility

### Description

Use each child's gradient-based influence on the parent's activation to distribute utility:

```
For each unit j with utility U_j, for each child k:
    g_{k->j} = f'(z_j) * w_{kj} * a_k       # gradient of a_j w.r.t. a_k, times a_k
    U_{k <- j} = U_j * g_{k->j} / sum_m |g_{m->j}|
```

This is a first-order Taylor approximation: "if child k's activation changed slightly, how much would parent j's activation change?"

### Pros
- **Preserves sign**: children with negative gradient-contributions get opposite-sign utility
- **Conserves utility** (via normalization)
- **O(1) per connection**
- **Naturally incorporates the activation derivative**: saturated parents automatically weight children less

### Cons
- **Gradients measure marginal sensitivity, not actual contribution.** This is the critical flaw. A child can have a large *current contribution* to a parent (it's the reason the parent has its current value) while having a gradient that points in the *opposite direction*. The gradient tells you what a small perturbation would do; it doesn't tell you what the child is actually doing.
- **Gives wrong answers in the harmful-feature case.** In the tested example, a child that is the primary *cause* of a parent's harmful activation gets *positive* utility, because its gradient opposes the parent's current value. The gradient correctly says "increasing this child would decrease the parent" but ignores that the child at its *current value* is what made the parent harmful.

### Why It Fails

The root cause is a distinction between **marginal** and **actual** contribution. Signed utility (Approach B) measures actual contribution: "what would the error be if this child's contribution were removed?" Gradient weighting measures marginal contribution: "what would happen if this child's value changed by epsilon?" These diverge when a child's contribution has a different sign than its gradient (e.g., a child contributes `c = -2.0` but its gradient is also negative, meaning increasing the child would push even more negative -- the gradient and the contribution both push the parent down, but the *utility formula* based on the gradient assigns opposite credit from what the actual-contribution formula assigns).

A variant using actual contributions instead of gradients (`g_{k->j} = c_{k->j}`) reduces to a signed version of Approach A, which has the same double-negative problem.

> Detailed failure analysis with worked example: [multilayer_signed_utility.md, Section 6](multilayer_signed_utility.md#6-approach-d-gradient-weighted-utility)

---

## Summary

| | Approach A | Approach B | Approach D |
|---|---|---|---|
| **Formula** | Proportional split | Signed utility + normalization | Gradient-weighted split |
| **Conserves utility** | Exact | Exact (via normalization) | Exact (via normalization) |
| **Correct signs (harmful case)** | No | Yes | No |
| **Same formula every layer** | No | Yes | No |
| **Activation derivative needed** | No | Yes | Yes |
| **O(1) per connection** | Yes | Yes | Yes |
| **Saturation behavior** | N/A | Degrades to signed-A | Gradients vanish |
| **Recommendation** | Baseline | **Primary candidate** | Not recommended |

## Recommended Plan

1. **Implement Approach B** as the primary algorithm
2. **Implement Approach A** as a simple baseline for comparison
3. Test both on the tracking task from `signed_utility_experiment.py` extended to multi-layer networks with sigmoid activations
4. Compare: do the correct sign assignments in B lead to better pruning/recycling decisions than A?

## Files

- `initial_notes.md` -- early brainstorming and problem framing
- `multilayer_signed_utility.md` -- detailed derivations, worked examples, and formal algorithm specification
