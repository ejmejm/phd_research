# Multi-Layer Signed Utility: Candidate Algorithms

## 1. Background: Signed Utility (Single Layer)

**Setup:** Linear prediction `y_hat = sum_i c_i` where `c_i = w_i * a_i`, target `y*`, error `e = y* - y_hat`.

**Formula:**
```
u_i = |e + c_i| - |e|
    = |error_without_i| - |error_with_all|
```

**Properties:**
1. `sum_i u_i <= |y*|` (finite utility budget per timestep)
2. Signed: `u_i > 0` means helpful, `u_i < 0` means harmful
3. Tracked as EMA: `U_i(t) = (1-d)*u_i(t) + d*U_i(t-1)`

**Goal for multi-layer:** Compute utility for every unit such that:
- Cross-layer comparison is meaningful (finite, shared budget)
- Sign is preserved (helpful vs harmful)
- O(1) per connection
- Same or similar formula at every layer

---

## 2. Test Network

```
Node 1 (input) ---w12---> Node 2 (hidden) ---w24---> Node 4 (output)
      \                        |                         ^
       \---w13---> Node 3 <---w23        w34------------/
                   (hidden) ---------------------------->
```

- Node 2: `z_2 = w_12 * a_1`, `a_2 = f(z_2)`
- Node 3: `z_3 = w_13 * a_1 + w_23 * a_2`, `a_3 = f(z_3)`
- Node 4: `z_4 = w_24 * a_2 + w_34 * a_3`, `y_hat = f(z_4)`

Processing order (reverse topological): Node 4 -> Node 3 -> Node 2 -> Node 1.

### Base Values (all-positive case)
- `a_1 = 2.0`, `y* = 5.0`
- `w_12 = 1.0, w_13 = 0.5, w_23 = 1.0, w_24 = 1.0, w_34 = 2.0`
- `f(x) = 0.5x`

Forward pass:
- `z_2 = 2.0, a_2 = 1.0`
- `z_3 = 1.0 + 1.0 = 2.0, a_3 = 1.0`
- `z_4 = 1.0 + 2.0 = 3.0, y_hat = 1.5`
- `e = 5.0 - 1.5 = 3.5`

### Harmful-Feature Values
- Same except `w_23 = -2.0, w_24 = 3.0` (node 2 helps output directly but hurts node 3)

Forward pass:
- `z_2 = 2.0, a_2 = 1.0`
- `z_3 = 1.0 - 2.0 = -1.0, a_3 = -0.5`
- `z_4 = 3.0 - 1.0 = 2.0, y_hat = 1.0`
- `e = 5.0 - 1.0 = 4.0`

---

## 3. Common First Step: Output-Layer Utility

All approaches share the same computation at the output layer. We work in pre-activation space:

```
z_out* = f_inv(y*)
e_z = z_out* - z_out
c_{j->out} = w_{j,out} * a_j

U_j^out = |e_z + c_{j->out}| - |e_z|
```

Property: `sum_j U_j^out <= |z_out*|`.

### Base case:
- `z_4* = f_inv(5.0) = 10.0`, `e_z4 = 7.0`
- `c_{2->4} = 1.0, c_{3->4} = 2.0`
- `U_2^out = |7+1| - |7| = 1.0`
- `U_3^out = |7+2| - |7| = 2.0`
- Sum = 3.0 <= 10.0. OK.

### Harmful case:
- `z_4* = 10.0`, `e_z4 = 8.0`
- `c_{2->4} = 3.0, c_{3->4} = -1.0`
- `U_2^out = |8+3| - |8| = 3.0`
- `U_3^out = |8-1| - |8| = -1.0` (harmful)
- Sum = 2.0 <= 10.0. OK.

---

## 4. Approach A: Proportional Redistribution

### Algorithm

For each unit `j` (processed in reverse topological order), distribute its total utility `U_j` to its children proportionally by contribution magnitude:

```
U_{k, from j} = U_j * |c_{k->j}| / sum_m |c_{m->j}|
```

A unit's total utility is the sum of what it receives from all parents:
```
U_k = sum_{j: k feeds j} U_{k, from j}
```

### Properties
- **Conserves utility exactly:** `sum_k U_{k,from j} = U_j`
- **Loses internal sign information:** All children get the same sign as the parent, regardless of whether they're helping or hurting that parent.
- **O(1) per connection.**
- **Simple to implement.**

### Base Case Example

**Output layer:** `U_2^out = 1.0, U_3^out = 2.0`

**Node 3** (utility 2.0) distributes to children:
- `c_{1->3} = 1.0, c_{2->3} = 1.0`, sum = 2.0
- `U_{1,from 3} = 2.0 * 1.0/2.0 = 1.0`
- `U_{2,from 3} = 2.0 * 1.0/2.0 = 1.0`

**Node 2** total utility: `U_2 = 1.0 (direct) + 1.0 (from 3) = 2.0`
- `c_{1->2} = 2.0` (only child)
- `U_{1,from 2} = 2.0`

**Final:** Node 3 = 2.0, Node 2 = 2.0, Node 1 = 1.0 + 2.0 = 3.0

Budget check: Output level total = 3.0. After redistribution through each node, same total flows to input. Total utility across all units = 2.0 + 2.0 + 3.0 = 7.0 (across 3 depth levels).

### Harmful Case Example

**Output layer:** `U_2^out = 3.0, U_3^out = -1.0`

**Node 3** (utility -1.0) distributes:
- `|c_{1->3}| = 1.0, |c_{2->3}| = 2.0`, sum = 3.0
- `U_{1,from 3} = -1.0 * 1/3 = -0.333`
- `U_{2,from 3} = -1.0 * 2/3 = -0.667`

**Problem:** Node 2 feeds into node 3 with `w_23 = -2.0`, making `c_{2->3} = -2.0`. Node 3 hurts the output because its activation is negative (`a_3 = -0.5`), and node 2 is the reason `a_3` is negative (it contributes -2.0 to `z_3`). So node 2 is the cause of node 3's harm. Yet Approach A gives node 2 a share of node 3's negative utility (-0.667), which is the right sign but for the wrong reason -- it's not because node 2 is identified as the cause of harm, it's just proportional to magnitude.

Conversely, node 1 contributes `c_{1->3} = 1.0` (positive, pushing `z_3` up, partially counteracting node 2's harmful push). Node 1 is actually *mitigating* node 3's harm. But it still gets negative utility (-0.333).

**Node 2** total: `3.0 + (-0.667) = 2.333`
- `U_{1,from 2} = 2.333`

**Final:** Node 3 = -1.0, Node 2 = 2.333, Node 1 = -0.333 + 2.333 = 2.0

**Approach A verdict:** Simple and conserves utility, but doesn't distinguish which children cause vs. mitigate harm. All children inherit the parent's sign.

---

## 5. Approach B: Recursive Signed Utility

### Algorithm

The idea: at each hidden unit, define a "pseudo-target" pre-activation, then apply the signed utility formula on the unit's inputs. This way, the same formula is used at every layer.

**Step 1 (output layer):** Same as above.

**Step 2 (hidden layers, reverse topological order):**

For unit `j` with total utility `U_j = sum_p U_{j->p}` (summed over all parents `p`):

1. Compute the target contribution to each parent. For parent `p`, node `j`'s contribution was `c_{j->p} = w_{jp} * a_j`. The ideal contribution would have been `c_{j->p}^* = c_{j->p} + U_{j->p}` (shifting by the utility, since that's how much more/less the parent's error would change). This gives ideal activation:
   ```
   a_j^* = a_j + sum_p (U_{j->p} / w_{jp})
   ```
   Wait, this doesn't work when j feeds multiple parents with different weights.

   **Simpler approach:** Define the pre-activation error for unit `j` based on its total utility:
   ```
   e_j = U_j / f'(z_j)
   ```
   This "inverts" the activation derivative to convert activation-space utility into pre-activation-space error.

2. Compute raw signed utilities for each child `k`:
   ```
   c_{k->j} = w_{kj} * a_k
   u_{k->j}^raw = |e_j + c_{k->j}| - |e_j|
   ```

3. **Normalize to conserve utility:**
   ```
   U_{k->j} = u_{k->j}^raw * |U_j| / sum_m |u_{m->j}^raw|
   ```
   This ensures `sum_k |U_{k->j}| = |U_j|` while preserving the signs from the formula.

4. Accumulate: `U_k = sum_j U_{k->j}`

### Why Normalization is Needed

Without normalization, when `U_j = 0`, we get `e_j = 0` and `u_{k->j}^raw = |c_{k->j}|` for each child -- potentially large utility from zero input utility. The normalization step ensures no utility is created or destroyed.

### The Sign Problem

When `U_j < 0` (unit is harmful), `e_j = U_j / f'(z_j) < 0`. This creates a pseudo-target that is *further in the harmful direction*. The signed utility formula then identifies children that push toward this bad target as "helpful" and children that push away as "harmful" -- the opposite of what we want.

**Proposed fix:** When `U_j < 0`, interpret the error as "the unit needs to move away from its current position by |U_j|/f'(z_j)." We flip the error direction:
```
e_j = |U_j| / f'(z_j)    (always positive)
```

Then children contributing in the *same direction as the unit's current activation* get negative utility (they're enabling the harm), and children opposing get positive utility (they're mitigating the harm). After computing raw utilities, multiply by `sign(U_j)` if we want to preserve the parent's sign, or leave as-is if we want child-level sign to reflect their own contribution.

Actually, let me think about this more carefully...

**Interpretation 1 (flip error for negative parents):**
```
e_j = -U_j / f'(z_j)   (positive when U_j < 0, meaning "move back toward neutral")
```
Children pushing toward neutral get positive raw utility. Then normalize by `|U_j|`.

**Interpretation 2 (error always means "how to improve"):**
For `U_j > 0`: target is `z_j + U_j/f'(z_j)` (do more of what you're doing)
For `U_j < 0`: target is `z_j - U_j/f'(z_j) = z_j + |U_j|/f'(z_j)` (undo what you're doing)

Wait, both interpretations give the same formula: `e_j = |U_j| / f'(z_j)` always positive? No...

Let me just be concrete. The question is: **what direction should z_j move to improve things?**

For `U_j > 0`: node is helpful. More contribution = better. `c_j = w_jp * a_j`. If `w_jp > 0`, increase `a_j` -> increase `z_j`. If `w_jp < 0`, decrease `a_j` -> decrease `z_j`. Direction: `sign(w_jp)`.

For `U_j < 0`: node is harmful. Less contribution = better. Opposite of above. Direction: `-sign(w_jp)`.

But `j` may feed multiple parents with different `w_jp`. In general, the direction is problem-dependent.

**Practical resolution: use the formula as-is (`e_j = U_j / f'(z_j)`) but always normalize.**

The raw formula gives signs that are meaningful *relative to the pseudo-target direction*. Even if the pseudo-target direction is "wrong" for negative-utility parents, the normalization ensures conservation. The raw signs still tell us which children are contributing in the same direction as the parent's contribution (whether that contribution is helpful or harmful).

Concretely: if parent `j` has `U_j = -1.0` and child `k` gets `u_{k->j}^raw = +2.0`, that means child `k` is pushing `j` in the direction `j` is going. Since `j` is going in a harmful direction, child `k` is enabling the harm. After normalization: `U_{k->j} = 2.0 * |-1.0| / S_raw > 0`. But this is *positive*, yet the child is enabling harm...

Hmm. There are two sign conventions we need to choose between:

**Convention 1: "How good is this child for the network?"**
- Child enabling a harmful parent -> negative utility
- Child mitigating a harmful parent -> positive utility
- Use `e_j = |U_j| / f'(z_j)` (always toward "improvement")

**Convention 2: "How much is this child contributing to the parent's behavior (good or bad)?"**
- Child enabling a harmful parent -> positive (it's causing the parent to behave as it does)
- Child mitigating a harmful parent -> negative
- Use `e_j = U_j / f'(z_j)` (toward "more of current behavior")
- Then multiply by `sign(U_j)` to convert to "good for network"

Both are valid. For pruning decisions (the end goal), Convention 1 is more useful: we want to know which units are net-positive for the network.

### Revised Algorithm (Convention 1)

```
e_j = |U_j| / f'(z_j)       # always positive: "how much improvement is needed"

For each child k:
    u_{k->j}^raw = |e_j + c_{k->j}| - |e_j|

# For units that are harmful (U_j < 0), flip signs:
# Children that enable the harm should get negative utility
if U_j < 0:
    u_{k->j}^raw = -u_{k->j}^raw

# Normalize to conserve |U_j|:
U_{k->j} = u_{k->j}^raw * |U_j| / sum_m |u_{m->j}^raw|
```

Wait, but when `U_j < 0` and we flip signs, children with large positive raw utility (those enabling the harmful direction) become negative. Children pushing against become positive. And `sum_k U_{k->j}` could be negative, which matches `U_j < 0`.

Actually we want `sum_k |U_{k->j}| = |U_j|` (conservation of budget magnitude). Let me just define:

```
U_{k->j} = u_{k->j}^raw * U_j / sum_m |u_{m->j}^raw|
```

No. Let me start from scratch with a cleaner formulation.

### Clean Algorithm

**For each hidden unit `j` with utility `U_j`, distributing to children:**

1. **Compute raw contribution scores** (how much each child contributes to `j`'s activation):
   ```
   e_j = |U_j| / f'(z_j)
   s_{k->j} = |e_j + c_{k->j}| - |e_j|     where c_{k->j} = w_{kj} * a_k
   ```

   `s_{k->j} > 0` means child `k` is contributing to `j`'s activation.
   `s_{k->j} < 0` means child `k` is working against `j`'s activation.

2. **Convert to utility** (account for whether `j`'s activation is good or bad):
   ```
   U_{k->j} = sign(U_j) * s_{k->j} * |U_j| / sum_m |s_{m->j}|
   ```

   - If `U_j > 0` (helpful parent): children contributing to `j`'s activation get positive utility.
   - If `U_j < 0` (harmful parent): children contributing to `j`'s activation get *negative* utility.

3. **Accumulate:** `U_k = sum_j U_{k->j}`

**Properties:**
- `sum_k |U_{k->j}| = |U_j|` (conservation)
- Sign reflects whether child is net-good for the network
- O(1) per connection
- Uses the signed utility formula at every layer (with an error defined by parent utility)

### Base Case Example

**Output:** `U_2^out = 1.0, U_3^out = 2.0` (same as before)

**Node 3** (`U_3 = 2.0 > 0`):
- `e_3 = |2.0| / 0.5 = 4.0`
- `c_{1->3} = 1.0, c_{2->3} = 1.0`
- `s_{1->3} = |4.0 + 1.0| - |4.0| = 1.0`
- `s_{2->3} = |4.0 + 1.0| - |4.0| = 1.0`
- Sum `|s| = 2.0`
- `U_{1->3} = (+1) * 1.0 * 2.0 / 2.0 = 1.0`
- `U_{2->3} = (+1) * 1.0 * 2.0 / 2.0 = 1.0`

**Node 2** (`U_2 = 1.0 + 1.0 = 2.0 > 0`):
- `e_2 = 2.0 / 0.5 = 4.0`
- `c_{1->2} = 2.0` (only child)
- `s_{1->2} = |4.0 + 2.0| - |4.0| = 2.0`
- `U_{1->2} = (+1) * 2.0 * 2.0 / 2.0 = 2.0`

**Final:** Node 3 = 2.0, Node 2 = 2.0, Node 1 = 1.0 + 2.0 = 3.0

Same as Approach A (expected: all contributions are positive, no sign differences to exploit).

### Harmful Case Example

**Output:** `U_2^out = 3.0, U_3^out = -1.0`

**Node 3** (`U_3 = -1.0 < 0`):
- `e_3 = |-1.0| / 0.5 = 2.0`
- `c_{1->3} = w_13 * a_1 = 0.5 * 2.0 = 1.0`
- `c_{2->3} = w_23 * a_2 = (-2.0) * 1.0 = -2.0`
- `s_{1->3} = |2.0 + 1.0| - |2.0| = 1.0` (node 1 contributes to node 3's activation)
- `s_{2->3} = |2.0 + (-2.0)| - |2.0| = 0 - 2.0 = -2.0` (node 2 opposes node 3's activation)
- Sum `|s| = 3.0`
- `sign(U_3) = -1`
- `U_{1->3} = (-1) * 1.0 * 1.0 / 3.0 = -0.333`
- `U_{2->3} = (-1) * (-2.0) * 1.0 / 3.0 = +0.667`

**Interpretation:** Node 3 is harmful to the output. Node 1 contributes positively to node 3's (harmful) activation, so node 1 gets negative utility via this path. Node 2 opposes node 3's activation (pushes it more negative, which actually made things worse in this case... wait).

Hmm, let me double-check. Node 3's pre-activation is `z_3 = -1.0`, activation `a_3 = -0.5`. Its contribution to the output is `c_{3->4} = w_34 * a_3 = 2.0 * (-0.5) = -1.0`. This is harmful because the output needs more positive contribution (error is positive).

Now, node 3 has `z_3 = -1.0` (negative). Node 2 contributes `c_{2->3} = -2.0` to `z_3`, which is what makes `z_3` so negative. Node 1 contributes `c_{1->3} = 1.0`, which is actually *reducing* how negative `z_3` is.

So: node 2 is the cause of node 3 being harmful (it pushes `z_3` very negative). Node 1 is partially mitigating the harm (it pushes `z_3` positive).

The algorithm gives: `U_{2->3} = +0.667` and `U_{1->3} = -0.333`.

That means node 2 gets **positive** utility through this path, even though it's the cause of the harm! And node 1 gets **negative** utility, even though it's mitigating!

**The issue:** `e_3 = 2.0 > 0` means "the target pre-activation is higher than current." Node 2 contributes `c_{2->3} = -2.0` which *opposes* the current activation direction (pushing more negative). The formula `s_{2->3} = -2.0` says node 2 is opposing the activation. Then `sign(U_3) = -1` flips the sign, giving node 2 positive utility.

The logic is: parent is harmful -> opposing the parent's activation is good -> node 2 opposes -> node 2 is good. But this logic is WRONG here because node 2 opposing the activation (pushing more negative) is exactly what makes the activation negative in the first place!

**Root cause:** The error direction `e_3 > 0` means "activation should be more positive." The signed utility formula with this error correctly identifies that `c_{1->3} = 1.0` (positive) is helping reach the target, and `c_{2->3} = -2.0` (negative) is opposing. But then the `sign(U_j)` flip reverses everything.

**The fix is to NOT flip signs.** The error `e_3 = |U_j|/f'(z_j) > 0` already encodes the correct direction (the unit needs more positive activation to be less harmful). Children helping achieve this should get positive utility.

### Revised Algorithm (Take 2)

```
e_j = |U_j| / f'(z_j)       # always positive

For each child k:
    s_{k->j} = |e_j + c_{k->j}| - |e_j|

# Normalize to conserve |U_j|:
U_{k->j} = s_{k->j} * |U_j| / sum_m |s_{m->j}|
```

No sign flip. The formula with the always-positive error naturally gives the right signs: children pushing toward improvement get positive utility, children pushing away get negative utility.

**But wait:** does `sum_k U_{k->j}` equal `U_j` or `|U_j|`? We're distributing `|U_j|` worth of utility, but the parent's utility could be negative.

For conservation, we want `sum_k |U_{k->j}| = |U_j|`. The normalization ensures this. But the *signed sum* `sum_k U_{k->j}` could differ from `U_j`.

This is actually fine for our purposes. The budget is conserved in absolute terms. The signed sum at each depth doesn't need to be preserved -- what matters is that the total magnitude of utility doesn't grow.

### Harmful Case (Revised, no sign flip)

**Node 3** (`U_3 = -1.0`):
- `e_3 = 1.0 / 0.5 = 2.0`
- `s_{1->3} = |2.0 + 1.0| - |2.0| = 1.0` (node 1 pushes toward higher z_3, which is the improvement direction)
- `s_{2->3} = |2.0 - 2.0| - |2.0| = -2.0` (node 2 pushes toward lower z_3, opposing improvement)
- Sum `|s| = 3.0`
- `U_{1->3} = 1.0 * 1.0 / 3.0 = +0.333`
- `U_{2->3} = (-2.0) * 1.0 / 3.0 = -0.667`

Node 1: positive utility via node 3 (it's pushing z_3 positive, which would reduce harm). Correct!
Node 2: negative utility via node 3 (it's pushing z_3 negative, causing the harm). Correct!

**Node 2** total: `U_2 = 3.0 + (-0.667) = 2.333`
- `e_2 = 2.333 / 0.5 = 4.667`
- `c_{1->2} = 2.0`
- `s_{1->2} = |4.667 + 2.0| - |4.667| = 2.0`
- `U_{1->2} = 2.0 * 2.333 / 2.0 = 2.333`

**Final:**
- Node 3: -1.0
- Node 2: 2.333
- Node 1: 0.333 + 2.333 = 2.667

**Budget check:** Output level: `|3.0| + |-1.0| = 4.0`. Node 3 distributes 1.0, node 2 distributes 2.333. At node 1: 2.667. The magnitudes don't grow beyond what was distributed. Conservation holds at each redistribution step: `|U_{1->3}| + |U_{2->3}| = 0.333 + 0.667 = 1.0 = |U_3|`. OK.

**Comparison with Approach A:** Approach A gave node 2 utility `2.333` (same) but node 1 utility `2.0` via the harmful-case path. Approach B gives node 1 utility `2.667`. The difference comes from node 1 getting **positive** credit for mitigating node 3's harm (Approach B), vs getting **negative** credit as a proportional share of a harmful parent (Approach A).

This is the key advantage: Approach B correctly identifies which children are helping vs hurting within each harmful parent.

---

## 6. Approach D: Gradient-Weighted Utility

### Algorithm

Use the gradient of each parent's utility with respect to child activations to determine how to distribute utility.

**Step 1 (output layer):** Same as above.

**Step 2 (hidden layers):**

For unit `j` with utility `U_j`, consider the gradient of the output error with respect to child activations. Specifically:

At the output, the error changes with respect to `a_j` via:
```
de/da_j = -dy_hat/da_j = -f'(z_out) * w_{j,out}
```

For a child `k` feeding unit `j`:
```
da_j/da_k = f'(z_j) * w_{kj}
```

The "importance" of child `k` to parent `j`:
```
g_{k->j} = da_j/da_k * a_k = f'(z_j) * w_{kj} * a_k
```

This is the child's contribution to the parent, scaled by the activation derivative.

Distribute utility:
```
U_{k->j} = U_j * g_{k->j} / sum_m |g_{m->j}|
```

**Note:** This is essentially a first-order Taylor approximation of "how much would removing child k's activation change parent j's activation?"

### Properties
- **Preserves sign:** `g_{k->j}` is signed, so children pushing in the wrong direction get opposite-sign utility.
- **Conserves utility:** By normalizing by `sum |g|`.
- **O(1) per connection** (just multiply weight * activation * derivative).
- **Issue:** Gradient magnitude != actual importance. A child at a steep part of the activation curve gets high gradient even if its actual contribution is small. Conversely, saturated activations suppress gradients even for important children.

### Base Case Example

**Output:** `U_2^out = 1.0, U_3^out = 2.0` (same)

**Node 3** (`U_3 = 2.0`):
- `f'(z_3) = 0.5` (constant for f(x) = 0.5x)
- `g_{1->3} = 0.5 * w_13 * a_1 = 0.5 * 0.5 * 2.0 = 0.5`
- `g_{2->3} = 0.5 * w_23 * a_2 = 0.5 * 1.0 * 1.0 = 0.5`
- Sum `|g| = 1.0`
- `U_{1->3} = 2.0 * 0.5 / 1.0 = 1.0`
- `U_{2->3} = 2.0 * 0.5 / 1.0 = 1.0`

**Node 2** (`U_2 = 1.0 + 1.0 = 2.0`):
- `g_{1->2} = 0.5 * 1.0 * 2.0 = 1.0`
- `U_{1->2} = 2.0 * 1.0/1.0 = 2.0`

**Final:** Node 3 = 2.0, Node 2 = 2.0, Node 1 = 1.0 + 2.0 = 3.0

Same as A and B for the all-positive linear case (expected).

### Harmful Case Example

**Output:** `U_2^out = 3.0, U_3^out = -1.0`

**Node 3** (`U_3 = -1.0`):
- `g_{1->3} = 0.5 * 0.5 * 2.0 = 0.5`
- `g_{2->3} = 0.5 * (-2.0) * 1.0 = -1.0`
- Sum `|g| = 1.5`
- `U_{1->3} = -1.0 * 0.5 / 1.5 = -0.333`
- `U_{2->3} = -1.0 * (-1.0) / 1.5 = +0.667`

**Interpretation:** Node 1 gets negative utility (it's contributing positively to a harmful node). Node 2 gets positive utility (it's working against the harmful node's... wait).

Node 2 has `g_{2->3} = -1.0` (negative gradient contribution). `U_{2->3} = -1.0 * (-1.0) / 1.5 = +0.667`.

But node 2 is what's making node 3 negative! `c_{2->3} = -2.0` pushes `z_3` very negative, which makes `a_3` negative, which hurts the output. The gradient says "node 2 pushes node 3's activation down" which in the context of `U_3 < 0` (harmful parent) means "node 2 makes the parent more harmful"... or does it?

Actually, `g_{k->j} = f'(z_j) * w_{kj} * a_k` measures the *signed effect* of child `k` on parent `j`'s activation.
- `g_{2->3} = 0.5 * (-2.0) * 1.0 = -1.0`: increasing node 2's activation would *decrease* node 3's activation.
- `g_{1->3} = 0.5 * 0.5 * 2.0 = 0.5`: increasing node 1's activation would *increase* node 3's activation.

The formula `U_{k->j} = U_j * g_{k->j} / sum |g|`:
- Node 1: `(-1.0) * 0.5 / 1.5 = -0.333`. Negative because the parent is harmful and node 1 helps the parent exist (positive gradient on harmful parent).
- Node 2: `(-1.0) * (-1.0) / 1.5 = +0.667`. Positive because it opposes the harmful parent.

**But this is wrong!** Node 2 is the primary reason node 3 is harmful. `c_{2->3} = -2.0` makes `z_3 = -1.0`, which makes `a_3 = -0.5`, which contributes -1.0 to the output. Node 2 is the cause, not the cure.

The issue: the gradient says "if node 2 increased slightly, node 3 would decrease." True, but node 2 at its *current value* is already what made node 3 so negative. The gradient measures *marginal* sensitivity, not *actual contribution*.

**Approach D gives the WRONG answer for this case.** It says node 2 is good (via node 3 path) when node 2 is actually the cause of node 3's harm.

### Approach D Corrected: Use Actual Contributions Instead of Gradients

Replace gradients with actual contributions:
```
g_{k->j} = c_{k->j} = w_{kj} * a_k
U_{k->j} = U_j * c_{k->j} / sum_m |c_{m->j}|
```

This is a *signed* version of Approach A. Let me redo:

**Node 3** (`U_3 = -1.0`):
- `c_{1->3} = 1.0, c_{2->3} = -2.0`, sum `|c| = 3.0`
- `U_{1->3} = -1.0 * 1.0 / 3.0 = -0.333`
- `U_{2->3} = -1.0 * (-2.0) / 3.0 = +0.667`

Same wrong answer! The issue is that `c_{2->3} = -2.0` (negative contribution) combined with `U_3 = -1.0` (harmful parent) double-negates to positive.

The problem: node 2 contributes *negatively* to node 3's pre-activation. Node 3's pre-activation being negative is what makes it harmful. So node 2 contributing negatively is what causes the harm. But mathematically, "contributing in the opposite direction of a harmful parent" looks like a double negative = positive.

**This reveals that the direction of "harm" depends on the relationship between the parent's activation and the output, not just the sign of the parent's utility.** Whether a child's contribution to a harmful parent is good or bad depends on which direction the parent needs to move to become helpful.

This is exactly what Approach B (revised, no sign flip) handles correctly, because it defines the error direction based on which way the activation needs to move.

---

## 7. Summary and Comparison

| Property | Approach A | Approach B (revised) | Approach D |
|----------|-----------|---------------------|-----------|
| Conserves utility | Yes (exact) | Yes (via normalization) | Yes (via normalization) |
| Signs correct for harmful case | No (all children get parent's sign) | Yes | No (gradient direction != contribution direction) |
| Formula at every layer | No (different formula for redistribution) | Yes (signed utility formula + normalization) | No (gradient-weighted formula) |
| O(1) per connection | Yes | Yes | Yes |
| Handles saturation | N/A | Potential issue (1/f'(z) blows up) | Same issue if using gradients |

**Recommendation:** Approach B (revised) is the most promising. It correctly identifies which children cause vs. mitigate harm, uses the same signed utility formula at every layer, and conserves utility. The main concern is behavior near activation saturation (addressed in Section 8).

Approach A is a reasonable simple baseline for comparison.

Approach D as formulated doesn't correctly handle the harmful-feature case.

---

## 8. Approach B: Saturation and Sigmoid

### Concern

For `f = sigmoid`, `f'(z) = sigma(z)(1-sigma(z))`. Near saturation (`z >> 0` or `z << 0`), `f'(z) -> 0`, so `e_j = |U_j|/f'(z_j) -> infinity`.

Is this actually a problem? Let's check: if `z_j` is heavily saturated and `U_j = 0.5`:
- `e_j = 0.5 / 0.001 = 500` (for example)
- `c_{k->j}` values are finite
- `s_{k->j} = |500 + c_k| - |500| ≈ c_k` for `c_k << 500`

So when `e_j` is very large relative to contributions, all children get `s_{k->j} ≈ c_k` (their raw contribution). This is actually reasonable: when there's a huge "error" to fix, every bit of positive contribution helps roughly equally.

The normalization then distributes `|U_j|` proportionally to `|c_k|`, with signs based on `sign(c_k)`. In the limit of large `e_j`, this converges to Approach A (proportional to magnitude, but with signs).

So **saturation is not a problem** -- it just makes Approach B converge to a signed version of Approach A, which is a graceful degradation.

### Sigmoid Example

Let's use the harmful-case network but with `f = sigmoid`.

- `a_1 = 2.0`, `w_12 = 1.0, w_13 = 0.5, w_23 = -2.0, w_24 = 3.0, w_34 = 2.0`, `y* = 0.9`

Forward pass:
- `z_2 = 1.0 * 2.0 = 2.0`, `a_2 = sigmoid(2.0) = 0.881`
- `z_3 = 0.5 * 2.0 + (-2.0) * 0.881 = 1.0 - 1.762 = -0.762`, `a_3 = sigmoid(-0.762) = 0.318`
- `z_4 = 3.0 * 0.881 + 2.0 * 0.318 = 2.643 + 0.636 = 3.279`, `y_hat = sigmoid(3.279) = 0.964`
- `e = 0.9 - 0.964 = -0.064`

Output layer:
- `z_4* = sigmoid_inv(0.9) = ln(0.9/0.1) = 2.197`
- `e_z4 = 2.197 - 3.279 = -1.082` (prediction too high, need to reduce z_4)
- `c_{2->4} = 3.0 * 0.881 = 2.643`
- `c_{3->4} = 2.0 * 0.318 = 0.636`
- `U_2^out = |-1.082 + 2.643| - |-1.082| = 1.561 - 1.082 = 0.479`
- `U_3^out = |-1.082 + 0.636| - |-1.082| = 0.446 - 1.082 = -0.636`

Node 2 is helpful (removing it would increase error). Node 3 is harmful (removing it would decrease error). This makes sense: the output is too high, and both nodes contribute positively to z_4, but node 2 contributes much more. Removing node 3 (smaller contribution) reduces z_4 less, getting closer to the target. Removing node 2 (larger) overshoots.

Wait, let me re-check. `e_z4 = -1.082` (negative). The signed utility `|e_z4 + c| - |e_z4|`:
- For node 2: `|-1.082 + 2.643| = |1.561| = 1.561`. `|-1.082| = 1.082`. Utility = 0.479 > 0.
- For node 3: `|-1.082 + 0.636| = |-0.446| = 0.446`. Utility = 0.446 - 1.082 = -0.636 < 0.

So removing node 2 takes us from error 1.082 to error 1.561 (worse) -- node 2 helps. Removing node 3 takes us from 1.082 to 0.446 (better) -- node 3 hurts. Makes sense!

**Approach B at node 3** (`U_3 = -0.636`):
- `f'(z_3) = sigmoid(-0.762) * (1 - sigmoid(-0.762)) = 0.318 * 0.682 = 0.217`
- `e_3 = |-0.636| / 0.217 = 2.930`
- `c_{1->3} = 0.5 * 2.0 = 1.0`
- `c_{2->3} = (-2.0) * 0.881 = -1.762`
- `s_{1->3} = |2.930 + 1.0| - |2.930| = 3.930 - 2.930 = 1.0`
- `s_{2->3} = |2.930 - 1.762| - |2.930| = 1.168 - 2.930 = -1.762`
- Sum |s| = 2.762
- `U_{1->3} = 1.0 * 0.636 / 2.762 = 0.230` (positive: node 1 pushes z_3 toward improvement)
- `U_{2->3} = (-1.762) * 0.636 / 2.762 = -0.406` (negative: node 2 pushes z_3 away from improvement)

Node 2 gets negative utility through this path because it's pushing node 3's pre-activation more negative, which makes node 3 more harmful. Correct!
Node 1 gets positive utility because it's pushing node 3's pre-activation more positive, mitigating harm. Correct!

**Node 2** total: `U_2 = 0.479 + (-0.406) = 0.073`
- `f'(z_2) = sigmoid(2.0) * (1-sigmoid(2.0)) = 0.881 * 0.119 = 0.105`
- `e_2 = |0.073| / 0.105 = 0.695`
- `c_{1->2} = 1.0 * 2.0 = 2.0`
- `s_{1->2} = |0.695 + 2.0| - |0.695| = 2.695 - 0.695 = 2.0`
- `U_{1->2} = 2.0 * 0.073 / 2.0 = 0.073`

**Final:**
- Node 3: -0.636
- Node 2: 0.073
- Node 1: 0.230 + 0.073 = 0.303

Budget: Output level `|0.479| + |-0.636| = 1.115`. Node 3 distributes `|0.230| + |0.406| = 0.636 = |U_3|`. OK. Node 2 distributes 0.073 = |U_2|. OK.

**This looks correct and well-behaved with sigmoid!** Node 2 ends up with low utility (0.073) because its positive direct contribution to the output (0.479) is mostly canceled by its negative effect through node 3 (-0.406). Node 1 is the most useful input (0.303), and node 3 is harmful (-0.636).

---

## 9. Final Algorithm: Approach B (Recursive Signed Utility)

### Formal Specification

**Input:** Network with activations `{a_j}`, pre-activations `{z_j}`, weights `{w_{kj}}`, activation function `f`, target `y*`.

**Output:** Utility `U_j` for every unit `j`.

```
# Step 1: Output layer
z_out* = f_inv(y*)
e_out = z_out* - z_out

For each unit j feeding the output:
    c_{j->out} = w_{j,out} * a_j
    U_{j->out} = |e_out + c_{j->out}| - |e_out|

# Step 2: Hidden layers (reverse topological order)
For each unit j (from output-adjacent to input-adjacent):
    U_j = sum over parents p: U_{j->p}

    # Compute error magnitude at j
    e_j = |U_j| / f'(z_j)

    # Compute raw scores for each child k
    For each child k feeding j:
        c_{k->j} = w_{kj} * a_k
        s_{k->j} = |e_j + c_{k->j}| - |e_j|

    # Normalize to conserve |U_j|
    S = sum over children k: |s_{k->j}|
    For each child k:
        U_{k->j} = s_{k->j} * |U_j| / S
```

**Edge cases:**
- If `S = 0` (all contributions are zero or perfectly cancel): distribute `|U_j|` uniformly with sign matching parent's sign. Or simply set all `U_{k->j} = 0` (no utility to distribute).
- If `f'(z_j) = 0` (fully saturated): `e_j -> infinity`, and `s_{k->j} -> c_{k->j}`. The method degrades to signed proportional redistribution.
- `f_inv(y*)` undefined (e.g., `y* = 0` or `y* = 1` for sigmoid): clip to a large finite value.

**Tracking over time (EMA):**
```
U_j(t) = (1 - decay) * u_j(t) + decay * U_j(t-1)
```

### Desired Properties (Verified)
1. **Finite budget:** At any depth, `sum_k |U_{k->j}| = |U_j|`. Total network utility <= `depth * |z_out*|`.
2. **Signed:** Children causing harm get negative utility; children mitigating harm get positive utility.
3. **Same formula at every layer** (signed utility formula + normalization).
4. **O(1) per connection.**
5. **Graceful degradation at saturation** (converges to signed proportional redistribution).
6. **Cross-layer comparable:** A unit at depth 1 and a unit at depth 5 with the same utility magnitude are equally important.
