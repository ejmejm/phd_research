# Experiment: Streaming iPC for TD Value Prediction

## Motivation

The Rotating MNIST experiments ([results](../analysis/sweep_comparison/results.md)) showed that streaming iPC works mechanically — persistent value nodes don't cause instability, T=1 is sufficient, and accuracy is competitive. But the benchmark couldn't distinguish genuine learning from inertia: the classification label was constant for 72 consecutive steps, making a trivial "repeat previous prediction" baseline nearly unbeatable (98.61%).

The deeper issue is about what streaming iPC is *for*. In the supervised classification setting, new input information at layer L needs multiple steps to propagate through the hidden layers to layer 0 before producing an updated prediction. With T=1, this propagation is incomplete. When the target doesn't change (as in Rotating MNIST), this doesn't matter — the output layer can maintain its prediction regardless of what the hidden layers are doing. But when the target changes every step, the hidden layers *must* carry useful, current representations for the output to be correct.

From a reinforcement learning perspective, the interesting setting is one where:

1. **You don't have access to immediate ground-truth labels.** In supervised learning, you clamp the true target and the network adjusts toward it. In RL, you only get a reward and a next state — the "correct" prediction is a discounted sum of future rewards that you never observe directly.

2. **Your predictions must change as the world changes.** In a non-stationary environment, the right answer at every step is different. "Be sticky" is not a strategy.

3. **Temporal correlation is naturally useful.** Consecutive states in an MDP are similar, so the representation from the previous state is informative about the current state. This is exactly the regime where persistent value nodes should provide genuine value — not as inertia, but as a warm start for a changing prediction.

4. **Prediction errors are about the future.** The TD error δ = r + γV(s') - V(s) is a prediction error about future outcomes, which aligns naturally with the predictive coding framework: the network's generative model predicts future returns, and prediction errors drive updates.


## Task: Random Walk Value Prediction

### Environment

A continuing random walk on a ring of N states (default N=20). At each step, the agent moves to one of its two neighbors with equal probability (0.5 left, 0.5 right). The ring wraps: state 0 and state N-1 are adjacent.

Each state has a fixed reward r(s). The default reward structure places a single reward at one state:

$$r(s) = \begin{cases} +1 & \text{if } s = 0 \\ 0 & \text{otherwise} \end{cases}$$

This produces a smooth value function V(s) that peaks at the rewarded state and decays with distance around the ring. The true value function is analytically computable, allowing exact measurement of prediction error.

### Network setup

The PC network is used as a function approximator mapping state features to a scalar value prediction:

- **Layer L (input):** Clamped to state features φ(s_t)
- **Layers 1 to L-1 (hidden):** Persistent value nodes (in streaming variant)
- **Layer 0 (output):** 1-dimensional, clamped to the TD target during training

### State features

Two feature representations, tested separately:

1. **One-hot** (N-dimensional): Each state gets a unique basis vector. The network has enough capacity to memorize V(s) for each state independently. This tests streaming iPC in the tabular-like regime where generalization between states is not required.

2. **Smooth features** (low-dimensional): Encode the state's position on the ring using sinusoidal features:

   $$\phi(s) = \left[\sin\left(\frac{2\pi k s}{N}\right),\; \cos\left(\frac{2\pi k s}{N}\right)\right]_{k=1}^{K}$$

   with K harmonics (e.g., K=4 gives 8-dimensional features). Nearby states have similar features, so the network must generalize. This is the more interesting case: persistent value nodes from a nearby state provide a genuinely informative warm start because the features (and the true values) are similar.


## Algorithm: TD(0) with Streaming iPC

At each timestep:

### Step 1: Observe state and predict

The agent is in state s_t. Clamp layer L to φ(s_t). The network's current value prediction is:

$$\hat{V}(s_t) = \mu^{(0)} = \theta^{(0)} \cdot f(x^{(1)})$$

This is read off **before any update**, using the persistent value nodes from the previous step. In the streaming variant, x^(1) still reflects the representation built up at s_{t-1} — but since s_t and s_{t-1} are neighbors on the ring, this representation is approximately correct.

### Step 2: Take action, observe reward and next state

The agent moves randomly: s_{t+1} ~ Uniform(neighbors(s_t)). Observe reward r_t = r(s_t).

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

For the BP baseline, the network is a standard MLP trained with online SGD. The loss is (V̂(s_t) - y_t)² and only V̂(s_t) is differentiated (semi-gradient TD, standard practice).


## Experimental Design

### Hyperparameters to sweep

| Parameter | Values | Notes |
|---|---|---|
| α (weight LR) | Powers of 2, range TBD | Expect different optimal range than Rotating MNIST |
| γ_inf (inference LR) | {0.1, 0.5, 1.0} | γ=0.5 was robust in Rotating MNIST |
| T (inference steps) | {1, 2, 4, 8} | T=1 is the streaming hypothesis |
| γ_td (discount factor) | 0.95 (fixed) | Standard value |
| N (ring size) | 20 (fixed initially) | Can vary later |
| Feature type | {one-hot, smooth (K=4)} | Test both regimes |
| Network depth | {3, 4} | Deeper = harder for T=1 |

### Phases

**Phase 1: LR tuning.** Sweep α for each variant with γ_inf=0.5, T=1, smooth features. Find the stable operating range.

**Phase 2: T and γ_inf sweeps.** At best α, sweep T and γ_inf to test whether T=1 is still sufficient in the TD setting (where targets change every step and are noisy).

**Phase 3: Feature comparison.** Compare one-hot vs smooth features. With one-hot features, value nodes from the previous state are *uninformative* (orthogonal features). With smooth features, they're informative. If streaming iPC's advantage disappears with one-hot features, that confirms the mechanism is about representational warm-starting, not just weight-update dynamics.

**Phase 4: Ring size and depth.** Scale up to test whether the approach holds with more states (weaker temporal correlation in feature space) and deeper networks (slower information propagation).

### Metrics

**Primary:**
- **Value error (MSVE):** Mean squared error between V̂(s) and V*(s) averaged over all states. Measures how well the network has learned the true value function. Compute periodically by evaluating V̂ at all states.
- **TD error (squared):** (r_t + γV̂(s_{t+1}) - V̂(s_t))². The online training signal. Should decrease as the network learns.
- **Energy:** Total free energy F = Σ||ε^(l)||². Measures internal model consistency.

**Secondary:**
- **Learning speed:** How many steps until MSVE drops below a threshold. Directly compares how fast each variant learns.
- **Per-state value error:** V̂(s) - V*(s) plotted for all states. Shows whether errors are uniform or concentrated (e.g., near the reward state where the value function changes fastest).
- **Value node similarity:** Cosine similarity between value nodes at consecutive steps. High similarity confirms that temporal correlation produces useful warm starts. Compare streaming vs forward-init to see how different the persistent state is from the reinitialized state.


## What to Look For

### The core question

Does streaming iPC learn the value function faster or more accurately than forward-init iPC, and is the advantage attributable to the persistent value nodes providing useful trajectory context?

### Predictions

**If streaming iPC works well here:** Persistent value nodes carry a representation of the local neighborhood on the ring, providing a warm start that reduces the inference computation needed at each step. The value function is learned faster than forward-init because each step starts from a better internal state. The advantage should be larger with smooth features (informative warm start) than with one-hot features (uninformative warm start).

**If streaming iPC struggles:** The noisy, non-stationary TD targets may interact badly with persistent value nodes. Unlike supervised learning where the target is ground truth, TD targets are bootstrap estimates that are wrong early in training. Persistent value nodes might propagate and amplify these early errors rather than providing useful warm starts. In this case, forward-init (which "forgets" bad value node states each step) might be more robust to bootstrap noise.

**If T=1 is no longer sufficient:** With targets changing every step, the hidden layers need to update their representations much faster than in the constant-label regime. T=1 may not propagate new information fast enough, and the advantage of streaming over forward-init may shrink or invert at T=1 while appearing at higher T.

### Diagnostic: the one-hot vs smooth features comparison

This is the cleanest test of the streaming hypothesis. With one-hot features, consecutive states have orthogonal representations — the value nodes from s_{t-1} contain zero useful information about s_t. Any advantage of streaming over forward-init must come from something other than representational warm-starting (e.g., weight-update dynamics). With smooth features, the warm start is informative. If the advantage appears only with smooth features, we can attribute it to the mechanism we care about.


## Implementation Notes

### Computing the true value function

For a random walk on a ring of N states with transition probability 0.5 to each neighbor and discount γ, the value function satisfies the Bellman equation:

$$V(s) = r(s) + \gamma \cdot \frac{1}{2}\left[V(s-1) + V(s+1)\right]$$

This is a circulant linear system that can be solved analytically via discrete Fourier transform, or numerically by solving the N×N linear system (I - γP)V = r where P is the transition matrix.

### Training loop structure

```
for step in range(total_steps):
    # 1. Observe s_t, compute prediction V̂(s_t) before update
    # 2. Take random action → s_{t+1}, observe r_t
    # 3. Compute TD target: y_t = r_t + γ * V̂(s_{t+1})
    # 4. Clamp layer 0 to y_t, run T iPC updates
    # 5. Periodically evaluate MSVE over all states
```

### Evaluation

Every eval_freq steps, sweep through all N states and compute V̂(s) for each (using a forward pass, without modifying persistent value nodes). Compare against V*(s) to get MSVE. This gives a clean measure of learning progress independent of the stochastic trajectory.
