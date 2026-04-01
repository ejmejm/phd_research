# Streaming Incremental Predictive Coding (Streaming iPC)

## Network Structure

A predictive coding network consists of $L+1$ layers of **value nodes** $x^{(0)}, x^{(1)}, \ldots, x^{(L)}$ connected by $L$ weight matrices $\theta^{(0)}, \theta^{(1)}, \ldots, \theta^{(L-1)}$.

The network follows a **generative (top-down) convention**:

- Layer $L$ (top): input layer, clamped to the observation $y_{\text{in}} \in \mathbb{R}^{784}$
- Layer $0$ (bottom): output layer, clamped to the target $y_{\text{out}} \in \mathbb{R}^{10}$
- Layers $1, \ldots, L-1$: hidden layers with persistent value nodes

Each weight matrix $\theta^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ generates a **top-down prediction** of layer $l$ from layer $l+1$:

$$\mu^{(l)} = \theta^{(l)} \cdot f(x^{(l+1)})$$

where $f$ is a nonlinear activation function (ReLU in this implementation).

The **prediction error** at each layer is:

$$\varepsilon^{(l)} = x^{(l)} - \mu^{(l)}$$

The **free energy** (total prediction error) is:

$$F = \sum_{l=0}^{L-1} \| \varepsilon^{(l)} \|^2$$


## Initialization

**Weights:** Each $\theta^{(l)}$ is initialized from a LeCun uniform distribution:

$$\theta^{(l)}_{ij} \sim \mathcal{U}\left(-\sqrt{3 / d_{l+1}},\; \sqrt{3 / d_{l+1}}\right)$$

where $d_{l+1}$ is the fan-in dimension.

**Value nodes:** All hidden value nodes are initialized to zero:

$$x^{(l)} = \mathbf{0} \in \mathbb{R}^{d_l}, \quad l = 1, \ldots, L-1$$

These value nodes **persist across observations** — they are never reset during training. This is the defining property of the streaming variant.


## Algorithm

At each timestep, the network receives a new observation $(y_{\text{in}}, y_{\text{out}})$ from the data stream and performs the following steps:

### Step 1: Clamp boundary layers

$$x^{(L)} \leftarrow y_{\text{in}}, \quad x^{(0)} \leftarrow y_{\text{out}}$$

Hidden value nodes $x^{(1)}, \ldots, x^{(L-1)}$ retain their values from the previous timestep.

### Step 2: Compute online prediction (before any update)

The network's current prediction is read off before weights or value nodes are modified:

$$\hat{y} = \arg\max\; \mu^{(0)} = \arg\max\; \theta^{(0)} \cdot f(x^{(1)})$$

This gives a true online accuracy metric — the prediction reflects only what the network knew before seeing the current target.

### Step 3: Run $T$ simultaneous iPC updates

For $t = 1, \ldots, T$:

**(a) Compute predictions and errors at all layers:**

$$\mu^{(l)} = \theta^{(l)} \cdot f(x^{(l+1)}), \quad l = 0, \ldots, L-1$$

$$\varepsilon^{(l)} = x^{(l)} - \mu^{(l)}, \quad l = 0, \ldots, L-1$$

**(b) Update hidden value nodes** (layers $1$ to $L-1$):

$$x^{(l)} \leftarrow x^{(l)} + \gamma \left( -\varepsilon^{(l)} + f'(x^{(l)}) \odot \left[ (\theta^{(l-1)})^\top \varepsilon^{(l-1)} \right] \right)$$

The two terms have clear interpretations:

- $-\varepsilon^{(l)}$: **top-down correction** — pushes $x^{(l)}$ toward the prediction from layer $l+1$
- $f'(x^{(l)}) \odot (\theta^{(l-1)})^\top \varepsilon^{(l-1)}$: **bottom-up correction** — propagates the error from layer $l-1$ upward through the weights

**(c) Update all weights:**

$$\theta^{(l)} \leftarrow \theta^{(l)} + \alpha \; \varepsilon^{(l)} \cdot f(x^{(l+1)})^\top, \quad l = 0, \ldots, L-1$$

This is a local Hebbian-like rule: each weight update is an outer product of the local prediction error and the presynaptic activation.

**Ordering note:** Within each iPC step, errors $\varepsilon^{(l)}$ are computed once from the current state (step a), then both the value node update (step b) and the weight update (step c) use these same errors. The updates are **simultaneous** — neither depends on the other's result within the same step. This is the key property of incremental PC (Salvatori et al., 2024) that distinguishes it from classical PC where inference and learning alternate.


## Variants

The algorithm above describes the **streaming** variant where value nodes simply persist. Two additional variants modify the value nodes between Step 1 (clamping) and Step 3 (iPC updates):

### Forward-init variant

After clamping, reinitialize all hidden value nodes via a top-down forward pass:

$$x^{(l)} \leftarrow \theta^{(l)} \cdot f(x^{(l+1)}), \quad l = L-1, L-2, \ldots, 1$$

This discards the persistent state and starts fresh from the network's current generative predictions. It is equivalent to standard (non-streaming) iPC applied one sample at a time.

### EMA variant

Blend the persistent value nodes with the forward-pass predictions:

$$x^{(l)}_{\text{fwd}} = \theta^{(l)} \cdot f(x^{(l+1)}), \quad l = L-1, \ldots, 1$$

$$x^{(l)} \leftarrow \beta \cdot x^{(l)}_{\text{old}} + (1 - \beta) \cdot x^{(l)}_{\text{fwd}}$$

where $\beta \in [0, 1]$ controls the blend. $\beta = 0$ recovers forward-init; $\beta = 1$ recovers pure streaming.


## Hyperparameters

| Symbol | Name | Description |
|--------|------|-------------|
| $\alpha$ | Weight learning rate | Step size for weight updates |
| $\gamma$ | Inference learning rate | Step size for value node updates |
| $T$ | Inference steps per observation | Number of simultaneous value + weight updates per input |
| $\beta$ | EMA blending coefficient | Value node persistence vs. forward-init (EMA variant only) |
| $L$ | Number of layers | Total layers including input and output |
| $d_l$ | Layer dimensions | Width of each layer |
| $f$ | Activation function | Applied to value nodes before computing predictions |


## Extension: TD(λ) for Value Prediction

The standard streaming iPC algorithm described above performs supervised learning: the target $y_{\text{out}}$ is clamped at layer 0 at each timestep. This section extends streaming iPC to **temporal-difference value prediction**, where the network learns to predict discounted future returns from a stream of observations and sparse rewards.

### Setup

The network predicts a scalar value $V(s_t)$ from observation $s_t$:

- Layer $L$: clamped to observation $s_t$
- Layer $0$: single output node, prediction $\hat{V}(s_t) = \theta^{(0)} \cdot f(x^{(1)})$
- Layers $1, \ldots, L-1$: persistent hidden value nodes (carry temporal memory)

The TD error for the transition from $s_{t-1}$ to $s_t$ with reward $r_{t-1}$ is:

$$\delta_{t-1} = r_{t-1} + \gamma_{\text{td}} \hat{V}(s_t) - \hat{V}(s_{t-1})$$

### Why standard iPC gives TD(0)

With a clamped TD target $x^{(0)} = r + \gamma_{\text{td}} \hat{V}(s')$ at the output, the prediction error at layer 0 is $\varepsilon^{(0)} = \delta$. At equilibrium, the iPC weight update at each layer factors as:

$$\varepsilon^{(l)} \cdot f(x^{(l+1)})^\top = \delta \cdot \nabla_{\theta^{(l)}} \hat{V}$$

So the standard Hebbian update $\theta^{(l)} \mathrel{+}= \alpha \cdot \varepsilon^{(l)} \cdot f(x^{(l+1)})^\top$ is exactly the semi-gradient TD(0) update. However, TD(0) propagates value backward through time only **one step per pass** through the data. For tasks with delayed rewards (e.g., a reward arriving many timesteps after the causal event), this requires many passes to propagate value to the relevant states.

### TD(λ) with eligibility traces

TD(λ) uses eligibility traces to assign credit across multiple timesteps in a single pass. The standard semi-gradient TD(λ) update is:

$$e_t = \gamma_{\text{td}} \lambda \, e_{t-1} + \nabla_\theta \hat{V}(s_t)$$

$$\theta \mathrel{+}= \alpha \, \delta_t \, e_t$$

This requires $\nabla_\theta \hat{V}$ — the gradient of the value function with respect to the weights — **separated from the TD error $\delta$**. In standard iPC, $\delta$ is baked into the prediction errors, so simply accumulating iPC Hebbian updates in a trace gives rescaled TD(0), not TD(λ).

### Computing $\nabla V$ locally via pipelined backward signaling

The gradient $\nabla_{\theta^{(l)}} \hat{V}$ can be computed via a backward signal that propagates one layer per timestep, preserving the local message-passing structure of iPC. Define a backward signal $b^{(l)}$ at each layer:

$$b^{(1)} = f'(x^{(1)}) \odot (\theta^{(0)})^\top$$

$$b^{(l)} = f'(x^{(l)}) \odot \left[ (\theta^{(l-1)})^\top \, b^{(l-1)} \right], \quad l = 2, \ldots, L-1$$

This mirrors the bottom-up error signal in standard iPC inference (which computes $f'(x^{(l)}) \odot (\theta^{(l-1)})^\top \varepsilon^{(l-1)}$), but with a **unit signal** at the output instead of the prediction error $\varepsilon^{(0)}$. Each layer only needs $b$ from its neighbor below and local quantities — the same communication pattern as standard iPC.

The gradient at each layer is then:

$$\nabla_{\theta^{(l)}} \hat{V} = b^{(l+1)} \cdot f(x^{(l+1)})^\top$$

with the convention $b^{(0)} = 1$ (so $\nabla_{\theta^{(0)}} \hat{V} = f(x^{(1)})^\top$ trivially).

For a 2-layer network ($L = 2$), $b$ reaches every layer in a single step. For deeper networks, $b$ can propagate one layer per timestep across successive observations, introducing a delay of $L-2$ timesteps at the deepest layer — a minor approximation when value nodes change slowly.

### Full algorithm

Persistent state per layer $l$: value nodes $x^{(l)}$, eligibility trace $e^{(l)}$, backward signal $b^{(l)}$.
Global persistent state: $V_{\text{old}}$, stored $\nabla V_{\text{old}}^{(l)}$ from previous step.

**Initialize:** Process $s_0$ — settle value nodes, compute $V_0$ and $\nabla V_0$. Set $e^{(l)} = \mathbf{0}$.

**For each observation** $(s_t, r_{t-1})$:

**Step 1 — Update eligibility traces** (local, per-layer):

$$e^{(l)} \leftarrow \gamma_{\text{td}} \lambda \, e^{(l)} + \nabla V_{\text{old}}^{(l)}, \quad l = 0, \ldots, L-1$$

**Step 2 — Settle value nodes** (standard iPC inference, local):

Clamp $x^{(L)} = s_t$ and $x^{(0)} = V_{\text{old}}$. For $t = 1, \ldots, T$:

$$\mu^{(l)} = \theta^{(l)} \cdot f(x^{(l+1)}), \quad \varepsilon^{(l)} = x^{(l)} - \mu^{(l)}$$

$$x^{(l)} \mathrel{+}= \gamma \left( -\varepsilon^{(l)} + f'(x^{(l)}) \odot (\theta^{(l-1)})^\top \varepsilon^{(l-1)} \right), \quad l = 1, \ldots, L-1$$

Weights are **not** updated during inference (value nodes only).

**Step 3 — Read value prediction:**

$$\hat{V}(s_t) = \theta^{(0)} \cdot f(x^{(1)})$$

**Step 4 — Compute TD error:**

$$\delta = r_{t-1} + \gamma_{\text{td}} \hat{V}(s_t) - V_{\text{old}}$$

**Step 5 — Update weights** (three-factor rule: global $\delta$ × local trace):

$$\theta^{(l)} \mathrel{+}= \alpha \, \delta \, e^{(l)}, \quad l = 0, \ldots, L-1$$

**Step 6 — Compute $\nabla V$ for next step** (one-layer backward signal, local):

$$\nabla_{\theta^{(0)}} \hat{V} = f(x^{(1)})^\top$$

$$\nabla_{\theta^{(l)}} \hat{V} = b^{(l+1)} \cdot f(x^{(l+1)})^\top, \quad l = 1, \ldots, L-1$$

where $b^{(l)}$ propagates one layer per timestep as defined above.

**Step 7 — Store:**

$$V_{\text{old}} \leftarrow \hat{V}(s_t), \quad \nabla V_{\text{old}}^{(l)} \leftarrow \nabla_{\theta^{(l)}} \hat{V}$$

Value nodes $x^{(1)}, \ldots, x^{(L-1)}$ persist to the next timestep (streaming).

### Relationship to standard iPC

| | Standard streaming iPC | Streaming iPC + TD(λ) |
|---|---|---|
| **Layer 0 clamping** | $x^{(0)} = y_{\text{target}}$ | $x^{(0)} = V_{\text{old}}$ |
| **Weight update** | $\alpha \, \varepsilon^{(l)} \cdot f(x^{(l+1)})^\top$ (local Hebbian) | $\alpha \, \delta \, e^{(l)}$ (global $\delta$ × local trace) |
| **Inference** | Simultaneous value + weight updates | Value nodes only (weights frozen) |
| **Temporal credit** | None (single-step) | Eligibility traces (multi-step) |
| **Communication** | One-layer message passing | Same + unit backward signal for $\nabla V$ |

With $\lambda = 0$, the eligibility trace reduces to $e^{(l)} = \nabla_{\theta^{(l)}} \hat{V}$ and the weight update becomes $\alpha \, \delta \, \nabla_{\theta^{(l)}} \hat{V}$, which recovers semi-gradient TD(0) — equivalent to standard iPC with a clamped TD target.

### Hyperparameters (additional)

| Symbol | Name | Description |
|--------|------|-------------|
| $\gamma_{\text{td}}$ | TD discount factor | Discount rate for future rewards |
| $\lambda$ | Trace decay | Eligibility trace decay rate; 0 = TD(0), 1 = Monte Carlo |
| $\alpha$ | Weight learning rate | Scales the TD error × trace weight update |


## Data Streams

### Rotating MNIST

A single MNIST digit is continuously rotated by $\delta$ degrees per timestep. After a full 360° rotation ($360 / \delta$ timesteps), a new random digit is drawn.

This produces temporally correlated inputs with periodic discontinuities (digit switches), testing both smooth tracking and recovery from abrupt changes.

### Audio Prediction Benchmark

A stream of musical chord audio observations (2500-dim binary FFT features) with sparse delayed scalar rewards. Three chord-reward associations must be learned: guitar C major → +1, piano C major → −1, piano D major → 0. Rewards arrive 0.5–2.0 seconds after the chord ends, creating a temporal credit assignment challenge that motivates the TD(λ) extension.
