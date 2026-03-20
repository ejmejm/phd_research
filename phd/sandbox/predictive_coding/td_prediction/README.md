# Experiment: Streaming iPC for TD Value Prediction

## Motivation

The Rotating MNIST experiments ([results](../analysis/sweep_comparison/results.md)) showed that streaming iPC works mechanically — persistent value nodes don't cause instability, T=1 is sufficient, and accuracy is competitive. But the benchmark couldn't distinguish genuine learning from inertia: the classification label was constant for 72 consecutive steps, making a trivial "repeat previous prediction" baseline nearly unbeatable (98.61%).

The deeper issue is about what streaming iPC is *for*. In the supervised classification setting, new input information at layer L needs multiple steps to propagate through the hidden layers to layer 0 before producing an updated prediction. With T=1, this propagation is incomplete. When the target doesn't change (as in Rotating MNIST), this doesn't matter — the output layer can maintain its prediction regardless of what the hidden layers are doing. But when the target changes every step, the hidden layers *must* carry useful, current representations for the output to be correct.

From a reinforcement learning perspective, the interesting setting is one where:

1. **You don't have access to immediate ground-truth labels.** In supervised learning, you clamp the true target and the network adjusts toward it. In RL, you only get a reward and a next state — the "correct" prediction is a discounted sum of future rewards that you never observe directly.

2. **Your predictions must change as the world changes.** In a non-stationary environment, the right answer at every step is different. "Be sticky" is not a strategy.

3. **Temporal correlation is naturally useful.** Consecutive states in an MDP are similar, so the representation from the previous state is informative about the current state. This is exactly the regime where persistent value nodes should provide genuine value — not as inertia, but as a warm start for a changing prediction.

4. **Prediction errors are about the future.** The TD error δ = r + γV(s') - V(s) is a prediction error about future outcomes, which aligns naturally with the predictive coding framework: the network's generative model predicts future returns, and prediction errors drive updates.


## Task: Continuous Grid World Value Prediction

### Environment

A continuing random walk in a continuous 2D space [0,1]² with rectangular barriers. At each step, the agent takes a random step:

$$s_{t+1} = s_t + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

If the step would cross a barrier or leave the boundary, the agent is reflected back (the component of motion perpendicular to the barrier/boundary is negated). This keeps the agent inside the valid region and produces a well-defined stationary distribution.

The step size σ controls temporal correlation between consecutive states. Small σ means consecutive states are very close in input space (strong warm-start signal); large σ weakens temporal correlation and makes streaming less advantageous.

The space contains rectangular barriers that block movement. Default layout (schematic, coordinates in [0,1]²):

```
+----------------------------+
|              |             |
|              |    R₁       |
|              |             |
|              |             |
|                            |
|                  |         |
|    R₂            |         |
|                  |         |
+----------------------------+
```

Two reward regions R₁ and R₂ are placed on opposite sides of the barriers. The reward function is:

$$r(s) = \begin{cases} +1 & \text{if } s \in R_1 \cup R_2 \\ 0 & \text{otherwise} \end{cases}$$

where R₁ and R₂ are small circular or rectangular regions centered at fixed locations.

This produces a value function with smooth gradients in open areas but sharp discontinuities at barriers — two points that are close in Euclidean distance but separated by a barrier have very different values, because the agent must take a long path around the barrier to reach the other side.

### Why this task requires nonlinearity

The state input is the continuous 2D position s = (x, y) ∈ [0,1]². A linear model on 2D input can only learn a plane through value space. But barriers create sharp discontinuities: points (0.37, 0.50) and (0.39, 0.50) may be close in Euclidean space but separated by a barrier, so their values differ substantially. The network must learn nonlinear features that encode barrier boundaries and path-based distances, not just Euclidean proximity.

Crucially, the continuous state space means the network cannot memorize individual states — it **must** generalize. This is a fundamental improvement over discrete grid cells or one-hot encodings, where a lookup table suffices.

### Why continuous position matters for streaming

With continuous positions, consecutive states s_t and s_{t+1} are close in input space by construction (separated by at most ~σ). This means:

- The hidden representation built for s_t is a genuinely useful warm start for s_{t+1} in the common case (no barrier between them)
- The network must generalize smoothly across the input space — it cannot memorize, so hidden representations must encode local structure
- The step size σ provides a direct experimental knob on temporal correlation strength

### Network setup

The PC network is used as a function approximator mapping continuous state coordinates to a scalar value prediction:

- **Layer L (input):** Clamped to state position φ(s_t) = (x, y)
- **Layers 1 to L-1 (hidden):** Persistent value nodes (in streaming variant), with nonlinear activations
- **Layer 0 (output):** 1-dimensional, clamped to the TD target during training


## Algorithm: TD(0) with Streaming iPC

At each timestep:

### Step 1: Observe state and predict

The agent is at position s_t. Clamp layer L to φ(s_t) = (x_t, y_t). The network's current value prediction is:

$$\hat{V}(s_t) = \mu^{(0)} = \theta^{(0)} \cdot f(x^{(1)})$$

This is read off **before any update**, using the persistent value nodes from the previous step. In the streaming variant, x^(1) still reflects the representation built up at s_{t-1} — but since s_t and s_{t-1} are close (||s_t - s_{t-1}|| ~ σ), this representation is approximately correct.

### Step 2: Take action, observe reward and next state

The agent moves randomly: s_{t+1} = reflect(s_t + ε), ε ~ N(0, σ²I). Observe reward r_t = r(s_t).

### Step 3: Compute TD target

Compute the bootstrap target using the network's prediction at the next state. To get V̂(s_{t+1}), temporarily clamp layer L to φ(s_{t+1}) and read μ^(0). This can either be done:
- **(a) Forward-pass estimate:** A single top-down forward pass θ^(0) · f(θ^(1) · f(... φ(s_{t+1}))) without modifying value nodes. Used only for target computation.
- **(b) Next-step prediction:** Wait until step t+1 and use the pre-update prediction from that step (one-step delay). More natural for streaming.

The TD target is:

$$y_t = r_t + \gamma \hat{V}(s_{t+1})$$

### Step 4: Train on TD target

Clamp layer 0 to y_t. Run T steps of iPC:

$$\varepsilon^{(l)} = x^{(l)} - \theta^{(l)} \cdot f(x^{(l+1)})$$

$$x^{(l)} \leftarrow x^{(l)} + \gamma_{\text{inf}} \left( -\varepsilon^{(l)} + f'(x^{(l)}) \odot (\theta^{(l-1)})^\top \varepsilon^{(l-1)} \right)$$

$$\theta^{(l)} \leftarrow \theta^{(l)} + \alpha \; \varepsilon^{(l)} \cdot f(x^{(l+1)})^\top$$

The value nodes persist to the next step (streaming variant) or are discarded (forward-init variant).


## Variants to Compare

| Variant | Value nodes | TD target method | Description |
|---|---|---|---|
| **Streaming iPC** | Persist | Next-step (b) | Core method. Value nodes carry trajectory context. |
| **Forward-init iPC** | Reinit each step | Forward-pass (a) | Standard iPC baseline. No memory between steps. |
| **EMA iPC** | Blended | Forward-pass (a) | Interpolation: x = β·x_old + (1-β)·x_fwd |
| **Online BP + SGD** | N/A | Forward-pass (a) | Standard TD(0) with backprop. Semi-gradient. |
| **Linear BP + SGD** | N/A | Forward-pass (a) | Linear model baseline. Demonstrates nonlinearity is needed. |

For the BP baseline, the network is a standard MLP trained with online SGD. The loss is (V̂(s_t) - y_t)² and only V̂(s_t) is differentiated (semi-gradient TD, standard practice).

For the linear baseline, V̂(s) = w · φ(s) + b — a single linear layer from the 2D input to a scalar output, trained with the same semi-gradient TD(0) update. This serves as a floor: it shows the best a linear model can achieve on this task, quantifying how much of the value function structure requires nonlinear features. Since the true value function has sharp discontinuities at barriers, the linear model can only fit a best-approximation plane, and the gap between the linear baseline and the nonlinear methods measures the nonlinearity demand of the task.


## Experimental Design

### Hyperparameters to sweep

| Parameter | Values | Notes |
|---|---|---|
| α (weight LR) | Powers of 2, range TBD | Expect different optimal range than Rotating MNIST |
| γ_inf (inference LR) | {0.1, 0.5, 1.0} | γ=0.5 was robust in Rotating MNIST |
| T (inference steps) | {1, 2, 4, 8} | T=1 is the streaming hypothesis |
| γ_td (discount factor) | 0.95 (fixed) | Standard value |
| σ (step size) | {0.01, 0.03, 0.1} | Controls temporal correlation |
| Barrier config | {default layout} | Can vary to stress-test streaming |
| Network depth | {3, 4} | Deeper = harder for T=1 |

### Phases

**Phase 1: LR tuning.** Sweep α for each variant with γ_inf=0.5, T=1, σ=0.03. Find the stable operating range.

**Phase 2: T and γ_inf sweeps.** At best α, sweep T and γ_inf to test whether T=1 is still sufficient in the TD setting (where targets change every step and are noisy).

**Phase 3: Step size variation.** Vary σ to directly control temporal correlation strength. Small σ (e.g., 0.01) means consecutive states are very similar — streaming should have a strong advantage from warm-starting. Large σ (e.g., 0.1) means consecutive states are far apart — warm starts are less informative and may even be misleading (more likely to cross a barrier). This traces out the streaming advantage as a function of temporal correlation, which is the core theoretical variable.

**Phase 4: Barrier density and depth.** Add more barriers to increase the frequency of misleading warm starts. Scale to deeper networks. Larger networks with more barriers test whether streaming's advantage is robust or fragile.

### Metrics

**Primary:**
- **Value error (MSVE):** Mean squared error between V̂(s) and V*(s) evaluated on a dense grid of points (e.g., 50×50 = 2500 evaluation points). Measures how well the network has learned the true value function.
- **TD error (squared):** (r_t + γV̂(s_{t+1}) - V̂(s_t))². The online training signal. Should decrease as the network learns.
- **Energy:** Total free energy F = Σ||ε^(l)||². Measures internal model consistency.

**Secondary:**
- **Learning speed:** How many steps until MSVE drops below a threshold. Directly compares how fast each variant learns.
- **Spatial error heatmap:** V̂(s) - V*(s) plotted as a heatmap over [0,1]². Shows whether errors are concentrated near barriers (where warm starts are misleading) or reward regions (where the value function changes fastest).
- **Value node similarity:** Cosine similarity between value nodes at consecutive steps. Compare streaming vs forward-init. Stratify by whether the transition crossed a barrier — expect high similarity for open-area transitions and low similarity for barrier crossings.
- **Barrier-crossing analysis:** Track prediction error separately for transitions that cross a barrier vs those that don't. If streaming iPC has higher error specifically after barrier crossings, this directly confirms the warm-start mechanism (good when informative, harmful when misleading).


## What to Look For

### The core question

Does streaming iPC learn the value function faster or more accurately than forward-init iPC, and is the advantage attributable to the persistent value nodes providing useful trajectory context?

### Predictions

**If streaming iPC works well here:** Persistent value nodes carry a representation of the local neighborhood, providing a warm start that reduces the inference computation needed at each step. The value function is learned faster than forward-init because each step starts from a better internal state. The advantage should be present for open-area transitions (informative warm start) but may be absent or negative for barrier-crossing transitions — with the net effect still positive because barrier crossings are the minority of transitions. The advantage should increase with smaller σ (stronger temporal correlation).

**If streaming iPC struggles:** The combination of noisy bootstrap TD targets and occasionally misleading warm starts (barrier crossings) may interact badly. Persistent value nodes might propagate errors from the wrong side of a barrier, and the bootstrap noise amplifies these errors rather than correcting them. Forward-init, which "forgets" every step, avoids this compounding.

**If T=1 is no longer sufficient:** With targets changing every step, the hidden layers need to update their representations much faster than in the constant-label regime. T=1 may not propagate new information fast enough, and the advantage of streaming over forward-init may shrink or invert at T=1 while appearing at higher T. This would be especially pronounced after barrier crossings, where the hidden representation needs a large update.

### Diagnostic: step size and barrier-crossing analysis

The step size σ provides a clean experimental knob on temporal correlation. Plotting streaming's advantage (MSVE_forward_init - MSVE_streaming) as a function of σ should show a monotonically decreasing curve: large advantage at small σ (strong temporal correlation, informative warm starts), diminishing to zero or negative at large σ (weak correlation, uninformative or misleading warm starts).

Barrier crossings provide the complementary within-trajectory diagnostic. By stratifying performance on transitions that do vs don't cross a barrier, we directly measure the effect of warm-start quality without confounds.


## Implementation Notes

### Computing the true value function

For the continuous random walk with Gaussian steps, reflected at barriers and boundaries, the value function satisfies the Bellman equation:

$$V(s) = r(s) + \gamma \int V(s') \, p(s'|s) \, ds'$$

where p(s'|s) is the reflected Gaussian transition density. This cannot be solved analytically, but can be approximated numerically:

1. **Discretize** the space onto a fine grid (e.g., 100×100), marking grid cells that overlap barriers as impassable.
2. **Approximate the transition matrix** P on this grid: for each cell, sample transitions from the Gaussian step model with reflection and compute the empirical distribution over destination cells.
3. **Solve** the linear system (I - γP)V = r on the grid.

Alternatively, use Monte Carlo rollouts: for each evaluation point, run many long trajectories and compute the discounted return average. The discretization approach is more efficient and deterministic.

The evaluation grid (e.g., 50×50 for MSVE computation) can be coarser than the grid used to solve for V* (e.g., 100×100).

### Training loop structure

```
for step in range(total_steps):
    # 1. Observe s_t = (x_t, y_t), compute prediction V̂(s_t) before update
    # 2. Take random step: s_{t+1} = reflect(s_t + ε), observe r_t = r(s_t)
    # 3. Compute TD target: y_t = r_t + γ * V̂(s_{t+1})
    # 4. Clamp layer 0 to y_t, run T iPC updates
    # 5. Periodically evaluate MSVE over evaluation grid
```

### Evaluation

Every eval_freq steps, evaluate V̂ on a fixed dense grid of points (e.g., 50×50 over [0,1]², excluding points inside barriers). Compare against precomputed V*(s) to get MSVE. This gives a clean measure of learning progress independent of the stochastic trajectory.

### Environment implementation

The continuous grid world needs to support:
- Configurable barrier placement (list of rectangular obstacles with (x, y, width, height))
- Configurable reward regions (circular or rectangular, with location and radius/size)
- Gaussian step with reflection at barriers and boundaries
- Efficient barrier intersection testing (for reflection and barrier-crossing detection)
- Precomputed true value function on a fine grid
- Visualization of value functions as heatmaps (true vs predicted, error maps)
- Detection of barrier crossings for diagnostic metrics (did the line segment from s_t to s_{t+1} intersect a barrier?)
