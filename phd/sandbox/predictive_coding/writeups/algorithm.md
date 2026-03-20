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


## Data Stream

The implementation uses **Rotating MNIST**: a single MNIST digit is continuously rotated by $\delta$ degrees per timestep. After a full 360° rotation ($360 / \delta$ timesteps), a new random digit is drawn.

This produces temporally correlated inputs with periodic discontinuities (digit switches), testing both smooth tracking and recovery from abrupt changes.
