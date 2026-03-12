# Extending Signed Utility to Multi-Layer Networks

## 1. Signed Utility in the Linear Case

### Setup

Consider a linear prediction: `y_hat = sum_i (w_i * x_i)`, with target `y*` and error `delta = y* - y_hat`.

The **contribution** of feature `i` is `c_i = w_i * x_i`, and the prediction is `y_hat = sum_i c_i`.

### Formula

The **signed utility** of feature `i` on a given timestep is:

```
U_i = |delta + c_i| - |delta|
    = |y* - y_hat + c_i| - |y* - y_hat|
    = |y* - sum_{j != i} c_j| - |y* - sum_j c_j|
```

In words: the absolute error *without* feature `i`'s contribution, minus the absolute error *with* all features. Positive utility means the feature is helping reduce the error; negative utility means it is actively hurting.

### Implementation References

- **Simple experiment** (`signed_utility_experiment.py`, line 60-61):
  ```python
  error_without = np.abs(delta + x[i] * w_idbd[i])
  utility = error_without - error_with
  ```
- **SignedCBPTracker** (`feature_recycling.py`, line 784):
  ```python
  utilities = torch.abs(target_error + contributions) - torch.abs(target_error)
  ```
  where `target_error = target - pred` and `contributions_i = activation_i * w_out_i`.

In both cases, utility is tracked as an exponential moving average:
```
U_i(t) = (1 - decay) * u_i(t) + decay * U_i(t-1)
```

### Properties

1. **Finite total utility.** On any single timestep, the sum of all utilities is bounded:
   ```
   sum_i u_i <= |y*|
   ```
   This follows from the triangle inequality. The total "budget" of utility that can be distributed across features is at most the magnitude of the target. Individual features can have utility greater than `|y*|` (if they are very helpful) or negative utility (if they are harmful), but the total is bounded.

2. **Signed and interpretable.** Unlike CBP utility (`|activation| * |outgoing_weight|`), signed utility distinguishes between features that help vs. hurt. A feature with large magnitude and large outgoing weights could have *negative* signed utility if it pushes the prediction in the wrong direction.

3. **Decomposition of error reduction.** Signed utility directly measures each feature's marginal contribution to reducing the absolute error. If you removed feature `i`, the error would increase by exactly `u_i` (when `u_i > 0`) or decrease by `|u_i|` (when `u_i < 0`).

4. **Conservation.** When the prediction is perfect (`delta = 0`), `u_i = |c_i|` for all features, and the total utility equals `sum_i |c_i|`. When prediction is bad, features that contribute in the right direction get credit, features in the wrong direction get penalized, and the total stays bounded by `|y*|`.


## 2. The Problem: Extending to Multi-Layer Networks

### Goal

Compute a utility for every feature (hidden unit) in a multi-layer network that:
- Retains the finite-budget property (allows cross-layer comparison)
- Is signed (distinguishes helpful vs. harmful features)
- Can be used for pruning/recycling decisions across layers on an equal footing

### Why the Linear Formula Doesn't Directly Apply

In the linear (single-layer) case, there is an implicit "ideal value" for each feature: the value that would minimize the loss. The signed utility formula works because we can express the error as a sum of independent contributions and ask "what happens if we remove one?"

For hidden units in a multi-layer network:

1. **No clear target for hidden units.** For the output layer, we have `y*`. For hidden units, there is no ground-truth target that a unit "should" produce. The value of a hidden unit is only meaningful in the context of how it's used downstream.

2. **Bounded activations.** With sigmoid or tanh activations, a hidden unit's output is constrained to a range (e.g., [0, 1] for sigmoid). Even if we could compute an "ideal" activation, it might be unreachable. In the linear case, there's no such constraint.

3. **Non-additive interactions.** In deeper networks, contributions are no longer simply additive. A hidden unit's value passes through nonlinearities before reaching the output, so the "remove this feature" counterfactual isn't as clean.

### Current State

The existing `SignedCBPTracker` only works for single hidden layer networks with one output target. It computes contributions at the last hidden layer as `activation * output_weight`, then applies the signed utility formula using the target directly. This approach cannot extend to deeper layers because there is no target to compare against at intermediate layers.


## 3. Proposed Approach: Recursive Error Back-Attribution

### Core Idea

Propagate the utility *backwards* through the network, layer by layer:

1. **Output layer:** Compute signed utility of each last-hidden-layer unit using the standard formula (we have the real target here).

2. **Hidden layers:** For each hidden unit that has signed utility `U_i` at depth `d`, distribute that utility to its "children" (the units at depth `d-1` that feed into it) based on how much each child contributed to making unit `i`'s activation useful.

This is analogous to backpropagation, but for utility rather than gradients.

### Sketch

At depth `d` (the output layer), for each unit `j` in the last hidden layer:
```
u_j^(d) = |error_without_j| - |error|      (standard signed utility)
```

At depth `d-1`, for each unit `k` feeding into hidden layer `d`:
- Unit `k` contributes to each unit `j` at depth `d` via `c_{k->j} = activation_k * w_{kj}`
- The "error" from `j`'s perspective is: how far is `j`'s activation from its "ideal"?
- Distribute `u_j^(d)` to `k` based on `k`'s relative contribution to `j`.

### Key Design Questions

**Q1: What is the "error" at a hidden unit?**

Options:
- **(a) Use the utility itself as a pseudo-target.** If unit `j` has utility `U_j = 3.0`, treat `3.0` as a proxy for "how much error-reduction is at stake" for `j`, and distribute it to children proportionally.
- **(b) Compute a counterfactual ideal activation.** For each hidden unit, compute what activation value would maximize the signed utility of the units it feeds. This is potentially expensive and may not have a closed-form solution with nonlinear activations.
- **(c) Use the pre-activation error.** Consider the pre-activation value `z_j = sum_k (w_{kj} * a_k) + b_j` and define a "target" pre-activation `z_j*` that would produce the ideal output activation. For sigmoid, `z_j* = sigmoid_inv(ideal_activation_j)`. Then apply the signed utility formula on pre-activations.

**Q2: How to distribute a parent's utility to its children?**

Options:
- **(a) Proportional to contribution magnitude.**
  ```
  u_k^(d-1) += U_j^(d) * |c_{k->j}| / sum_m |c_{m->j}|
  ```
  Simple, but always distributes positive shares even when a child is harmful to the parent.
- **(b) Apply signed utility formula recursively.**
  Treat the parent's pre-activation as the "prediction" and compute signed utility of each child's contribution to it, then scale by the parent's utility. This preserves the sign property.
- **(c) Gradient-weighted attribution.** Use `|du_j / da_k|` or `du_j / da_k` to weight how much child `k` matters for parent `j`.

**Q3: Does the finite-budget property hold?**

If at each depth level, we redistribute exactly the same total utility that was computed at the level above, then:
- Total utility at any depth = total utility at the output layer <= `|y*|`
- Total utility across all depths <= `depth * |y*|`

This gives the **maximum total utility in the network = depth * |y*|**, which is the property you wanted. This makes utility comparable across layers: a unit at depth 1 with utility 2.0 is as "valuable" as a unit at depth 5 with utility 2.0, because both are measured against the same budget.

For this to work, the redistribution must be **conservative** -- the utility distributed to children of a parent must sum to exactly the parent's utility (or its absolute value if we want to preserve signs).

### A Concrete Proposal (Option B2: Recursive Signed Utility)

For each unit `j` at depth `d` with utility `U_j`:

1. Compute the pre-activation: `z_j = sum_k (w_{kj} * a_k) + b_j`
2. Define the "target" pre-activation: `z_j* = z_j + adjustment` where `adjustment` is derived from `U_j` and the activation function's derivative.
   - Simpler alternative: just use `z_j` as the "prediction" and treat `|U_j|` as the magnitude of the "target" for redistribution purposes.
3. For each child `k`, compute `c_{k->j} = w_{kj} * a_k`
4. Compute child `k`'s utility relative to parent `j`:
   ```
   u_{k,j} = |z_j - sum_{m != k} c_{m->j} - b_j| - |z_j - sum_m c_{m->j} - b_j|
           = |c_k| - |0|    (this simplifies trivially since z_j = sum_m c_{m->j} + b_j!)
   ```

   This doesn't work because the "error" at the hidden unit is zero by definition (the pre-activation is exactly what it is). We need a notion of what the pre-activation *should* be.

### The Fundamental Issue and Possible Resolution

The signed utility formula requires a notion of "error" -- the gap between what *is* and what *should be*. At the output, this is clear: `y* - y_hat`. At hidden layers, there is no natural target.

**Possible resolution: Propagate error, not utility.**

Instead of trying to define targets for hidden units, propagate the *output error* backward:

1. At the output: `e = y* - y_hat`. Each output-layer unit `j` has signed utility `u_j`.
2. For unit `j` with utility `u_j > 0`: it is helping. The "error it's responsible for correcting" is approximately `u_j`. Pass this back to its children.
3. For unit `j` with utility `u_j < 0`: it is hurting. The "error it's responsible for creating" is `|u_j|`. Pass this back to its children (as a signal that they should be blamed/credited for this harm).

At each child `k` of parent `j`, define:
```
pseudo_error_j = u_j                   (how much error unit j is addressing/creating)
c_{k->j} = w_{kj} * a_k               (child k's contribution to parent j)
pred_j = sum_m c_{m->j}                (total input to j, excluding bias)
partial_error_j = pseudo_error_j - pred_j  ... (this doesn't make dimensional sense)
```

This is where it gets tricky and multiple approaches diverge. The document above captures the problem space and the key design questions. The next step is to prototype and test these options.


## 4. Candidate Approaches to Evaluate

### Approach A: Proportional Redistribution (Simple Baseline)
```
u_k = sum_j [ U_j * (|c_{k->j}| / sum_m |c_{m->j}|) ]
```
- Always conserves utility (sum of children's utilities = parent's utility)
- Loses sign information at redistribution step (all children get positive share of a positive-utility parent)
- Simple to implement, good baseline

### Approach B: Recursive Signed Utility with Pseudo-Targets
For each parent `j` with utility `U_j`, define a pseudo-target for its pre-activation:
```
z_j_target = f_inv(a_j + U_j * da_j/dy)   (where f is the activation function)
```
Then apply signed utility formula on children relative to this pseudo-target.
- Preserves sign property
- Requires invertible activation function (or approximation)
- Needs careful handling of saturation regions

### Approach C: Shapley-Value Inspired
Compute exact marginal contribution of each child to each parent's utility by evaluating counterfactuals (what if child `k` were zero?).
- Most theoretically principled
- Expensive (O(children) forward passes per parent per step)
- Could be approximated

### Approach D: Gradient-Weighted Utility
```
u_k = sum_j [ U_j * (dU_j/da_k) * a_k / sum_m |(dU_j/da_m) * a_m| ]
```
- Uses existing gradient infrastructure
- Naturally captures nonlinear interactions
- May not perfectly conserve utility without normalization


## 5. Next Steps

1. Formalize which properties are *must-have* vs. *nice-to-have*
2. Work through a small 2-layer sigmoid example by hand for each approach
3. Implement the most promising approach(es)
4. Test on the tracking task from `signed_utility_experiment.py` extended to multi-layer
