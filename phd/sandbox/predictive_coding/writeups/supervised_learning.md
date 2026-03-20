# Experiment: Streaming Incremental Predictive Coding

## Motivation

Incremental Predictive Coding (iPC) (Salvatori et al., ICLR 2024) updates value nodes and weights simultaneously at every inference timestep, removing the need for separate E-step/M-step scheduling. In standard mini-batch training, a forward-pass initialization is still needed each time the batch changes, costing O(L) sequential operations and breaking the O(1) per-step elegance.

**Core hypothesis:** In a streaming setting where inputs change gradually (temporally correlated observations), the value nodes from the previous timestep serve as a warm start for the current input. This eliminates the need for forward-pass reinitialization and enables true single-step (T=1) or few-step (T=small) operation per new observation, while still learning effectively.

This is motivated by:
- Ororbia's SpNCN work showing streaming PC is viable with spiking neurons
- PCN-TA (IROS 2025) showing that preserving latent states across temporally correlated frames reduces required inference steps by 50%
- Hybrid Predictive Coding showing that after sufficient training, a single feedforward sweep suffices for familiar data
- The biological argument that brains don't clamp stimuli and wait for convergence

## Algorithm: Streaming iPC

Below is the core algorithm. The key difference from standard iPC is that value nodes `x^(l)` are **never reinitialized** — they persist across input changes.

```
Initialize:
  Weights θ^(l) for l = 0, ..., L-1 (random)
  Value nodes x^(l) for l = 1, ..., L-1 (zeros or small random)

For each observation (y_in, y_out) from the stream:
  Clamp x^(L) = y_in, x^(0) = y_out   # supervised case

  For t = 1, ..., T:   # T can be as low as 1
    # Compute predictions and errors (all layers in parallel)
    For each layer l:
      μ^(l) = θ^(l) · f(x^(l+1))
      ε^(l) = x^(l) - μ^(l)

    # Update value nodes (all layers in parallel)
    For each hidden layer l = 1, ..., L-1:
      x^(l) += γ · (-ε^(l) + f'(x^(l)) * θ^(l-1)^T · ε^(l-1))

    # Update weights (all layers in parallel)
    For each layer l = 0, ..., L-1:
      θ^(l) += α · ε^(l) · f(x^(l+1))^T
```

### Variants to test

1. **Streaming iPC (T=1):** Single inference + weight update step per observation. Maximum throughput, most aggressive.

2. **Streaming iPC (T=k):** k inference + weight update steps per observation. Test k ∈ {1, 2, 4, 8} to find the sweet spot.

3. **Streaming iPC + forward init:** On each new observation, do a forward pass to reinitialize value nodes, then run T steps of iPC. This is the standard mini-batch approach applied per-sample. Serves as an upper-bound baseline for the streaming variants.

4. **Streaming iPC + partial forward init:** Only reinitialize the layer(s) closest to the clamped ends (layers 1 and L-1), leave interior layers warm. A compromise.

5. **Streaming iPC + value node momentum/EMA:** Instead of raw persistence, blend old value nodes with forward-pass predictions: `x^(l) ← β · x^(l)_old + (1-β) · μ^(l)_forward`. Test β ∈ {0.0, 0.5, 0.9, 1.0} where β=0 is full reinit and β=1 is pure persistence.

## Experimental Setup

### Task 1: Rotating MNIST (primary)

**Data stream:** Take a single MNIST digit and continuously rotate it by a small angle δ per step. The stream cycles through angles 0° → 360°. After completing a rotation, switch to the next digit. This gives temporally correlated inputs with periodic discontinuities (digit switches).

- Input: 28×28 flattened image at layer L
- Output: one-hot label at layer 0
- Rotation speed: δ ∈ {1°, 5°, 15°} per step (controls temporal correlation strength)
- Test both within-digit tracking (does accuracy stay high during rotation?) and cross-digit transitions (how fast does it recover after a digit switch?)

**Why this task:** Simple enough to iterate quickly, but has the key properties: gradual change with occasional discontinuities, classification objective, well-understood baseline performance.

### Task 2: Sequential CIFAR-10 frames (secondary)

**Data stream:** Sort CIFAR-10 images by visual similarity (e.g., via embedding distance from a pretrained model, or simply group by class and use augmentation interpolation). Present images in this sorted order so consecutive frames are visually similar.

This is harder and tests whether the approach scales beyond toy settings.

### Task 3: Moving MNIST (stretch goal)

**Data stream:** Two digits bouncing around a 64×64 frame. The network sees one frame per step. This is a standard video prediction benchmark adapted for streaming classification/reconstruction.

### Architecture

Fully connected MLP for initial experiments:
- Layers: L ∈ {3, 4, 5} (i.e., 1-3 hidden layers)
- Hidden dim: 256 or 512
- Activation: ReLU (standard in PC literature, f' is simple)

The generative direction runs top-down: layer L (input) → layer 0 (output).

### Baselines

1. **Standard iPC (mini-batch):** Standard iPC with forward init per batch. Same architecture, same data (but shuffled into mini-batches). This is the "what you'd normally do" baseline.

2. **Online SGD/BP:** Standard backprop, one gradient step per observation, same streaming order. This controls for whether any learning algorithm can handle the streaming regime.

3. **Online SGD/BP (shuffled):** Same as above but with shuffled (i.i.d.) data order. This is the "ideal conditions" upper bound for standard training.

4. **Standard PC (mini-batch):** Original PC algorithm with T=20 inference steps then one weight update. Shows the cost of the full inference phase.

### Hyperparameters

- Weight learning rate α: sweep {1e-4, 5e-4, 1e-3, 5e-3}
- Inference learning rate γ: sweep {0.1, 0.5, 1.0} (literature suggests 0.5 is robust)
- Value node persistence β (for EMA variant): {0.0, 0.5, 0.9, 1.0}
- Inference steps per observation T: {1, 2, 4, 8}
- Network depth L: {3, 4, 5}
- Hidden dim: {256, 512}

### Metrics to Track

**Primary:**
- **Online accuracy:** classification accuracy computed on each observation *before* the weight update for that observation. This is the true online metric.
- **Energy F:** total free energy (sum of squared prediction errors across layers). Track per-step.
- **Layer-wise prediction error:** ‖ε^(l)‖ per layer, per step. Critical for understanding if errors propagate properly in the streaming regime.

**Secondary:**
- **Recovery time after discontinuity:** when the digit switches, how many steps until accuracy recovers to within 90% of steady-state? This directly measures the cost of not reinitializing.
- **Value node drift:** ‖x^(l)_t - μ^(l)_t‖ averaged over time. If value nodes track predictions well, streaming works. If they diverge, it doesn't.
- **Weight update magnitude:** ‖Δθ^(l)‖ per layer. Are deeper layers still getting meaningful updates?
- **Comparison of weight trajectories:** do streaming iPC weights converge to similar solutions as batch iPC?

**Diagnostics:**
- **Effective T:** in the EMA variant, how many steps after an input change does it take for value nodes to be within ε of where a full forward init would put them?
- **Gradient alignment:** cosine similarity between the streaming iPC weight update and the BP gradient for the same input. Measures how much the stale value nodes degrade the learning signal.

## Implementation Notes

Use JAX with Equinox. The PC update equations are simple enough to implement directly without autograd — the gradients of the free energy w.r.t. x and θ have closed-form expressions (Eqs. 6 and 7 from the paper). This is actually one of the appeals of PC: no computational graph needed.

### Key implementation details

- **Layer indexing:** Follow the paper's convention where L = input, 0 = output. This is the generative model direction. Document this clearly in the code.
- **Value node state:** Store as a list/tuple of arrays, one per hidden layer. These persist across observations in the streaming variants and are part of the scan carry.
- **Clamping:** x^(L) and x^(0) are never updated by inference — they're set from the data each step.
- **Batching over data:** In the streaming setting, there's only one observation at a time (batch_size=1). For the mini-batch baselines, use standard batching.
- **The weight update** (Eq. 7) is a local Hebbian-like outer product: Δθ^(l) = α · ε^(l) · f(x^(l+1))^T. This is embarrassingly parallel across layers.
- **Track all metrics inside the scan** for efficiency. Log in the outer Python loop.

### Pseudocode for the streaming train step

```python
class StreamState(eqx.Module):
    weights: list[Array]        # θ^(l) for each layer
    value_nodes: list[Array]    # x^(l) for hidden layers (persistent!)
    step: Array

def streaming_ipc_step(state, observation):
    y_in, y_out = observation

    # Clamp input and output layers
    x = [y_out] + list(state.value_nodes) + [y_in]  # x^(0), ..., x^(L)

    for t in range(T):
        # Compute predictions and errors
        predictions = [θ @ f(x[l+1]) for l, θ in enumerate(state.weights)]
        errors = [x[l] - predictions[l] for l in range(L+1)]

        # Update hidden value nodes (layers 1..L-1)
        for l in range(1, L):
            x[l] = x[l] + γ * (-errors[l] + f'(x[l]) * (state.weights[l-1].T @ errors[l-1]))

        # Update weights (all layers)
        new_weights = [
            θ + α * jnp.outer(errors[l], f(x[l+1]))
            for l, θ in enumerate(state.weights)
        ]
        state = state._replace(weights=new_weights)

    # Accuracy computed BEFORE updates would be the true online metric,
    # so compute it at the start of the function in practice
    new_value_nodes = x[1:-1]  # extract hidden layers
    return StreamState(weights=new_weights, value_nodes=new_value_nodes, step=state.step+1), metrics
```

## Expected Outcomes and What to Look For

**Optimistic scenario:** Streaming iPC with T=1 works well when inputs change slowly (small δ), with accuracy close to batch iPC. Value nodes track smoothly. Recovery after discontinuities takes a few steps but is fast. The EMA variant with β≈0.9 provides a good balance.

**Pessimistic scenario:** T=1 gives degraded learning signals because errors don't propagate to deeper layers within a single step. Deeper networks suffer more. Weight updates for early layers (near input, high l) are fine but later layers (near output, low l) get stale error signals.

**Key diagnostic:** If layer-wise prediction errors show that errors at deeper layers are consistently stale or misaligned, that confirms the concern about insufficient error propagation. The solution space then includes:
- Larger T (obvious but expensive)
- Shallower networks (fewer layers to propagate through)
- Layer-wise learning rate scaling (higher α for layers farther from the clamped ends)
- The EMA partial-reinit compromise

**What would be a publishable result:** Demonstrating that streaming iPC with small T (1-4) achieves comparable accuracy to batch iPC on temporally correlated streams, with a clear characterization of when it works and when it breaks (as a function of temporal correlation, network depth, and T). Connecting this to the continual learning literature (especially the no-task-boundary setting) would strengthen the contribution.

## Related Work to Cite

- Salvatori et al. (2024) — iPC (the base algorithm)
- Ororbia (2023) — Spiking Neural Predictive Coding for streaming
- Tschantz et al. (2023) — Hybrid Predictive Coding (amortized + iterative)
- PCN-TA (IROS 2025) — Temporal amortization in PC
- Ernoult et al. (2020) — Equilibrium Propagation with continual weight updates (parallel E/M steps, similar spirit)
- Song et al. (2020) — Z-IL / BP equivalence in PC
- Ororbia et al. (2020) — Continual learning with predictive coding
- Neal & Hinton (1998) — Incremental EM (theoretical foundation)
- Goemaere et al. (2024) — Error optimization in deep PC (addressing signal decay in deep networks, relevant to the depth-scaling concern)