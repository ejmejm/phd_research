# Multi-Layer Signed Utility: Edge Cases and Desired Outcomes

This document defines key edge cases for propagating utility through hidden layers, and specifies what the correct behavior should be in each case (and why).

All cases use the same setup: parent unit $j$ with utility $U_j$, children $k$ with contributions $c_k = w_{kj} \cdot a_k$, pre-activation $z_j = \sum_k c_k$, activation $a_j = f(z_j)$. We compute a pseudo-error $e_j = |U_j| / f'(z_j)$ and raw scores $s_k = |e_j + c_k| - |e_j|$.

---

## 1. Canceling Contributions, Nonlinear Activation

**Setup:** Sigmoid unit with $c_A = +1$, $c_B = -1$. So $z_j = 0$, $a_j = \sigma(0) = 0.5$. Parent receives $U_j = 2.0$ from its own parents.

**Pseudo-error:** $e_j = 2.0 / \sigma'(0) = 2.0 / 0.25 = 8.0$

**Raw scores:** $s_A = |8 + 1| - 8 = 1.0$, $s_B = |8 - 1| - 8 = -1.0$

**Desired outcome:** $U_A = +1.0$, $U_B = -1.0$, sum $= 0$.

**Why:** The children's contributions cancel — node A pushes $z$ positive, node B pushes it negative, and the net pre-activation is zero. The unit's output of 0.5 (and the utility it generates) comes entirely from the sigmoid's behavior at zero, which acts like a bias. Children should not receive credit for utility they didn't create. Node A still deserves positive utility (it pushes toward a higher activation, which is the improvement direction) and node B deserves negative utility (it opposes improvement). But their magnitudes should not be inflated to sum to $U_j = 2$.

**What goes wrong with normalization:** Any approach that forces $\sum |U_{k \leftarrow j}| = |U_j|$ would scale these scores up to $\pm 1$ with sum $= 2$ (approach B) or try to divide by $\sum s_k = 0$ and blow up (approach C). Both are wrong — the first over-credits children, the second is undefined.

---

## 2. Canceling Contributions, Non-smooth Activation

**Setup:** LTU with $c_A = +1$, $c_B = -0.99$. So $z_j = 0.01$, $a_j = 1$. Parent receives $U_j = 2.0$ from its own parents.

**Desired outcome:** $U_A = +1.0$, $U_B = -0.99$, sum $= 0.01$.

**Why:** The children's contributions nearly cancel. The step function did almost all the work — it took a tiny net input of $0.01$ and amplified it to $1.0$. The activation function "created" $1.99$ out of $2.0$ units of utility. Children collectively receive only $0.01$ — proportional to the actual signal they produced after cancellation.

Node A gets slightly more positive credit than B gets negative, because A is the one tipping $z_j$ above the threshold. But neither deserves much — the step function's discontinuity is doing the heavy lifting.

---

## 3. Zero Utility Parent

**Setup:** Linear activation $f(x) = x$ with $c_A = +1$, $c_B = -1$. So $z_j = 0$, $a_j = 0$. Parent receives $U_j = 0$.

**Desired outcome:** Depends on the output weight of the parent. If the output weight is non-zero, then the reason for the 0 utility is the 0 parent value. If the output weight is zero, then there is nothing to do. In the former case the normal utility function should be used, and in the latter case all children get $U_{k \leftarrow j} = 0$. An important thing to realize here is that that utility should not be the target, the target should be what the output value should be to minimize the loss. You can have 0 utility, and the input weights can still have the ability to affect the main error if the output weight is non-zero. But at the same time, if there is 0 utility to spread, then do we want to split that into negative and positive utility? That could result in a whole isolated part of the network getting units with very high utility and very low utility that actually cannot have an effect on the main error.



# Normal Cases

## 1. Aligned Contributions, Linear Activation

**Setup:** Linear activation $f(x) = x$ with $c_A = +3$, $c_B = +2$. So $z_j = 5$, $a_j = 5$. Parent receives $U_j = 4.0$.

**Pseudo-error:** $e_j = 4.0 / 1.0 = 4.0$

**Raw scores:** $s_A = |4 + 3| - 4 = 3.0$, $s_B = |4 + 2| - 4 = 2.0$

**Desired outcome:** $U_A = 3.0$, $U_B = 2.0$, sum $= 5.0$.

**Why:** Both children push in the same direction, and the linear activation adds nothing — it just passes through $z_j$. The sum of scores (5.0) exceeds $U_j$ (4.0), so we cap: scale $= \min(1, 4/5) = 0.8$. Final: $U_A = 2.4$, $U_B = 1.6$, sum $= 4.0$.

**Why cap:** Without the cap, children would receive more total utility than the parent has. For a linear activation, the unit is a pure pass-through — all utility comes from children, none from the activation function. The cap ensures children collectively receive at most $|U_j|$.

---

## 2. Large Cancellation, Nonlinear Activation

**Setup:** Sigmoid unit with $c_A = +10$, $c_B = -9$. So $z_j = 1$, $a_j = \sigma(1) \approx 0.731$. Parent receives $U_j = 0.5$.

**Pseudo-error:** $f'(1) = \sigma(1)(1 - \sigma(1)) \approx 0.197$. $e_j = 0.5 / 0.197 \approx 2.54$.

**Raw scores:** $s_A = |2.54 + 10| - 2.54 = 10.0$, $s_B = |2.54 - 9| - 2.54 = 6.46 - 2.54 = 3.92$

**Desired outcome:** Both positive (both push toward improvement direction given the error sign). $\sum |s_k| = 13.92 \gg 0.5 = |U_j|$. Cap: scale $= 0.5 / 13.92 = 0.036$. Children get $U_A = 0.36$, $U_B = 0.14$.

**Why:** The contributions are large but mostly cancel ($z_j = 1$ from inputs summing to $\pm 19$). The sigmoid compresses this to $a_j = 0.731$, and the utility is only 0.5. Children did a lot of "work" that the activation function compressed away. The cap ensures we don't attribute more utility than the parent actually has.

---

## 3. Harmful Parent, Opposing Children

**Setup (from main derivation):** Sigmoid unit with $c_A = +1.0$ (from node 1), $c_B = -2.0$ (from node 2). $z_j = -1.0$, $a_j = \sigma(-1) \approx 0.269$. Unit is harmful: $U_j = -1.0$.

**Pseudo-error:** $f'(-1) \approx 0.197$. $e_j = 1.0 / 0.197 \approx 5.08$.

**Raw scores:** $s_A = |5.08 + 1| - 5.08 = 1.0$, $s_B = |5.08 - 2| - 5.08 = -1.92$ (approximately)

Wait — $s_B = |5.08 + (-2)| - 5.08 = 3.08 - 5.08 = -2.0$.

$\sum |s_k| = 3.0$. $|U_j| = 1.0$. Scale $= \min(1, 1/3) = 1/3$.

**Desired outcome:** $U_A = 1.0 \cdot 1/3 = +0.333$, $U_B = -2.0 \cdot 1/3 = -0.667$.

**Why:** The parent is harmful ($U_j < 0$). Node A pushes $z_j$ positive (toward improvement — less negative activation, less harm), so it gets positive utility. Node B pushes $z_j$ negative (causing the harm), so it gets negative utility. The cap applies because the combined magnitude of children's scores (3.0) exceeds the parent's utility magnitude (1.0).

**Sign correctness:** This is the critical test case. Node B is the *cause* of the parent's harm (its $-2.0$ contribution makes $z_j$ negative), and it correctly receives negative utility. Node A *mitigates* the harm, and correctly receives positive utility. Approaches that lose sign information (A) or invert signs (D, gradient-based) fail this case.

---

## 4. Saturated Activation

**Setup:** Sigmoid unit with $z_j = 6$, $a_j = \sigma(6) \approx 0.998$. $c_A = 4$, $c_B = 2$. $U_j = 0.3$.

**Pseudo-error:** $f'(6) \approx 0.0025$. $e_j = 0.3 / 0.0025 = 120$.

**Raw scores:** $s_A = |120 + 4| - 120 = 4.0$, $s_B = |120 + 2| - 120 = 2.0$.

$\sum |s_k| = 6.0 \gg 0.3$. Scale $= 0.3 / 6.0 = 0.05$.

**Desired outcome:** $U_A = 0.2$, $U_B = 0.1$. Sum $= 0.3 = U_j$.

**Why:** When the activation is saturated, $f'(z_j) \approx 0$, making $e_j$ very large. In this regime, scores converge to $s_k \approx c_k$ (the linear/proportional regime). The cap kicks in and distributes $|U_j|$ proportionally to contribution magnitudes. This is graceful degradation — it reduces to a signed version of Approach A, which is the best we can do when the activation function is nearly flat and provides no useful curvature information.

---

## Summary: The Activation Function as Utility Source/Sink

The key principle across these cases: **the activation function can absorb utility but children cannot create it.**

- When children's raw scores sum to less than $|U_j|$ (cases 1, partially 4): the activation function is responsible for the difference. This happens when contributions cancel but $f(0) \neq 0$, or when the activation function's nonlinearity amplifies small inputs.

- When children's raw scores sum to more than $|U_j|$ (cases 2, 3, 5): the activation function compresses the children's work. The cap prevents over-attribution.

The formula:
$$U_{k \leftarrow j} = s_k \cdot \min\!\left(1,\; \frac{|U_j|}{\textstyle\sum_m |s_m|}\right)$$

handles all these cases uniformly. It requires no special-casing for activation function type, saturation, or cancellation.
