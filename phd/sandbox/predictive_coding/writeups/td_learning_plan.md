# Implementation Plan: TD(0) Value Prediction with Streaming iPC

## Why This Experiment

The Rotating MNIST results ([results](../analysis/sweep_comparison/results.md)) identified four confounds that prevent us from concluding streaming iPC's advantage comes from meaningful learning:

1. **"Be sticky" is nearly optimal.** The label is constant for 72 steps at delta=5deg. A trivial repeat-previous-prediction baseline achieves 98.61%, higher than all methods except streaming iPC (98.76%). The interesting variation lives in a 1.4% sliver.

2. **Output layer can solve the task alone.** When the label doesn't change for dozens of steps, the output layer can maintain its prediction without the hidden layers tracking input changes.

3. **Information propagation is untested.** With T=1, new input at layer L propagates one layer per step. In a 5-layer network, it takes ~4 steps to reach the output. With a constant label, this latency is irrelevant.

4. **Streaming's advantage may be smoothing, not learning.** Persistent value nodes that don't change much look like high accuracy when not changing IS the correct strategy.

TD value prediction on a continuous random walk resolves all four confounds: the target changes every step, the network must generalize across continuous space, there's no ground-truth label, and barriers create sharp discontinuities that demand nonlinear representations.


## What We're Testing

### Primary hypothesis

Streaming iPC's persistent value nodes provide a **warm-start advantage**: when consecutive states are similar, the representation from s_t is a useful initialization for s_{t+1}, reducing inference computation and leading to faster/better value function learning.

### Three experiments, in order of importance

#### Experiment 1: Step size sigma variation

The cleanest test of the warm-start hypothesis. sigma controls temporal correlation directly:
- Small sigma -> consecutive states very similar -> warm start highly informative
- Large sigma -> states far apart -> warm start less useful or misleading

**Prediction:** Streaming's MSVE advantage over forward-init decreases monotonically with sigma.

**Why this is the best test:** It isolates temporal correlation as the independent variable. If streaming were just better for some other reason (implicit regularization, different optimization dynamics), the advantage wouldn't depend on sigma.

**Values:** sigma in {0.005, 0.01, 0.03, 0.1}

#### Experiment 2: Barrier-crossing stratification

The mechanistic smoking gun. Two points close in Euclidean space but separated by a barrier have very different values. A warm start from one side of a barrier is actively misleading on the other side.

**Prediction:** Streaming has lower TD error than forward-init on open-area transitions, but higher error on barrier-crossing transitions. Net effect positive because crossings are rare.

**Why this matters:** If streaming were just "better optimization" generically, it would be better everywhere. Being specifically worse at barrier crossings is a direct fingerprint of the warm-start mechanism.

**Implementation:** The `barrier_crossing` flag is computed during CPU-side trajectory sampling (line segment vs barrier intersection test) and logged as a per-step metric. TD errors are aggregated separately for barrier-crossing vs open-area transitions.

#### Experiment 3: T sensitivity

In Rotating MNIST, T=1 was sufficient because the label was constant. Here, the TD target changes every step.

**Prediction:** T=1 still works for streaming (persistent nodes compensate for limited inference) but not for forward-init (must rebuild representations from scratch each step). If streaming at T=1 matches forward-init at T=4, that's strong evidence that persistent nodes substitute for inference iterations.

**Values:** T in {1, 2, 4}

### What we're NOT testing

- **Architecture sweeps:** Proven unimportant in Rotating MNIST. Fixed at 3 layers, 64 hidden.
- **EMA variant tuning:** Include in LR sweep for completeness, but it's an interpolation between the two things we're comparing, not the core question.
- **Barrier density/depth:** Stretch goal if primary results are interesting.
- **Different reward layouts:** Fixed layout suffices to demonstrate the mechanism.


## What We're Plotting

### Figure 1: Value function heatmaps

5-panel: True V*, Streaming V-hat, Forward-init V-hat, Streaming error, Forward-init error. Barrier locations overlaid as rectangles, reward regions marked.

**Purpose:** Makes the task concrete. Shows WHERE errors concentrate (near barriers? near rewards?). Visually compelling for any audience.

### Figure 2: MSVE learning curves by sigma

Subplot per sigma value, lines per variant (streaming, forward-init, BP), 95% CI from 5 seeds.

**Purpose:** The main quantitative evidence. Shows convergence speed AND final quality at each correlation strength. Subplot structure directly visualizes how the comparison changes with sigma.

### Figure 3: Streaming advantage vs sigma

X-axis: sigma. Y-axis: MSVE(forward_init) - MSVE(streaming) at best LR per condition. Error bars from 5 seeds.

**Purpose:** THE figure for the paper. Directly tests the core prediction. A monotonically decreasing curve is strong evidence for the warm-start hypothesis. A flat line would refute it.

### Figure 4: Barrier-crossing stratified TD error

Two panels: "Open-area transitions" vs "Barrier-crossing transitions." Learning curves of mean TD error per category, lines per variant.

**Purpose:** The within-trajectory mechanistic decomposition. Streaming specifically worse after barrier crossings but better otherwise = direct fingerprint of warm-start mechanism.

### Figure 5: T sensitivity

Learning curves for T in {1, 2, 4} at best LR, one panel per variant.

**Purpose:** Does streaming at T=1 match forward-init at higher T? If yes, persistent nodes substitute for inference iterations.

### Figure 6: LR sensitivity

Standard sensitivity plot per variant. Methodological, not a main result.

### What we're NOT plotting

- **Energy/free energy:** Less interpretable with noisy TD targets. MSVE is the direct measure of value function quality.
- **Value node cosine similarity:** Interesting mechanistically but secondary. Add later if results warrant.
- **Per-layer error breakdowns:** Too detailed for the main story.


## Implementation

### New files (all under `phd/sandbox/predictive_coding/`)

| File | Description |
|------|-------------|
| `environment.py` | Continuous grid world: barriers, reward regions, Gaussian steps with reflection, trajectory sampling |
| `value_function.py` | True V* via discretized Bellman solve; MSVE evaluation on grid |
| `td_train.py` | TD(0) training loop for all variants (Hydra entry point) |
| `conf/td_config.yaml` | Base Hydra config for TD experiments |
| `conf/td_sweeps/` | Sweep YAML configs for mlflow-sweeper |
| `analysis/td_value_prediction/td_analysis.ipynb` | Analysis notebook |

### Reused code (no modifications needed)

| Module | What we use |
|--------|-------------|
| `models.py` | `PCNetwork`, `init_pc_network`, `pc_forward_pass`, `ipc_step` — all work with 1D output |
| `phd/jax_core/models.py` | `MLP` class for BP baseline |
| `phd/jax_core/utils.py` | `configure_jax`, `count_params`, `stack_pytrees`, `tree_replace` |
| `phd/research_utils/logging.py` | `init_experiment`, `log_metrics`, `log_child_metrics`, etc. |
| `phd/feature_search/jax_core/experiment_helpers.py` | `prepare_optimizer`, `set_seed`, `rng_from_string` |
| `phd/research_utils/analysis/plotting.py` | `plot_learning_curves`, `plot_param_sensitivity` |
| `phd/research_utils/analysis/analysis_utils.py` | `load_experiment_data`, `get_best_ablation_values` |

### Key technical decisions

**1D scalar output.** Network layer dims: `(1, 64, 64, 2)` — output=1, input=2. The existing `ipc_step` and `pc_forward_pass` work with arbitrary dimensions.

**Forward-pass V-hat(s') for TD target (method a from td_learning.md).** A full top-down pass with s' clamped, ignoring streaming value nodes. This makes the TD target identical across all variants given the same weights, isolating the effect of value node initialization on the main prediction. For BP, the forward pass is the only option anyway.

**MSVE evaluation always uses forward-pass predictions.** Even for the streaming variant, MSVE is computed by running `pc_forward_pass` at each grid point. This measures how well the *weights* have learned V*, independent of the current streaming state. Fair comparison: streaming's value nodes at a random trajectory position tell you nothing about V at arbitrary grid points.

**Environment stepping on CPU, TD targets in JAX.** Trajectories are pre-sampled on CPU (cheap NumPy random walks) and shipped to JAX as `(s_t, s_{t+1}, reward, barrier_crossing)` arrays. V-hat(s') must be computed inside the JIT'd scan since it depends on current weights.

**hidden_dim=64, activation=tanh.** With 2D input (not 784D), the network is tiny (~4.3K params). Tanh is bounded and smooth, standard for value function approximation, avoids dead neurons with small input dimension.

**5 seeds per config.** TD learning with function approximation is high-variance due to bootstrap targets. 5 seeds gives tighter confidence intervals, critical for the sigma sweep where we need to detect potentially small differences.


### Environment design

Continuous 2D grid world [0,1]^2 with two offset vertical barriers creating a maze-like structure:

```
+----------------------------+
|              |             |
|              |    R1       |
|              |             |
|              |             |
|                            |
|                  |         |
|    R2            |         |
|                  |         |
+----------------------------+
```

Agent takes Gaussian steps with reflection at barriers and boundaries. The barriers create a value function with smooth gradients in open areas but sharp discontinuities at barrier edges — two points close in Euclidean distance but separated by a barrier have very different values because reaching one from the other requires a long detour.

Reward is +1 inside circular reward regions R1 (upper-right) and R2 (lower-left), 0 elsewhere. Discount gamma=0.95.


### Training loop structure

```
Precompute: V* on 100x100 grid (once at startup, ~1s)
Precompute: eval grid points (50x50, excluding barrier interiors)

for scan in range(num_scans):
    # CPU: pre-sample trajectory chunk
    positions, rewards, barrier_crossings = grid_world.sample_trajectory(log_freq, seed)
    # Ship to JAX: (s_t, s_{t+1}, reward, barrier_crossing)
    observations = (positions[:-1], positions[1:], rewards, barrier_crossings)

    # JIT'd vmapped scan over seeds
    train_state, metrics = vmapped_scan(train_state, observations)

    # Log per-chunk metrics
    log_metrics({'td_error': ..., 'td_error_open': ..., 'td_error_barrier': ...})

    # Periodic MSVE evaluation
    if step % eval_freq == 0:
        msve = evaluate_msve(network, eval_grid, v_star)
        log_metrics({'msve': msve})
```


### Sweep phases

#### Phase 1: LR tuning (4 sweeps, 20 grid points)

Find stable operating range for each variant before comparing them.

| Sweep | Variant | alpha values | Grid points |
|-------|---------|-------------|-------------|
| `streaming_ipc_lr` | streaming iPC | 2^{-18, -16, -14, -12, -10} | 5 |
| `forward_init_ipc_lr` | forward-init iPC | 2^{-12, -10, -8, -6, -4} | 5 |
| `bp_lr` | BP (MLP+SGD) | 2^{-12, -10, -8, -6, -4} | 5 |
| `linear_bp_lr` | Linear (SGD) | 2^{-10, -8, -6, -4, -2} | 5 |

Fixed: sigma=0.03, T=1, gamma_inf=0.5, 3 layers, 64 hidden, tanh. 5 seeds atomic.

#### Phase 2: sigma sweep (2 sweeps, 24 grid points)

The key experiment. At best LRs from Phase 1, sweep sigma for the two main variants.

| Sweep | Variant | sigma values | alpha values | Grid points |
|-------|---------|-------------|-------------|-------------|
| `streaming_sigma` | streaming | {0.005, 0.01, 0.03, 0.1} | best 3 from P1 | 12 |
| `forward_init_sigma` | forward-init | {0.005, 0.01, 0.03, 0.1} | best 3 from P1 | 12 |

#### Phase 3: T sweep (1 sweep, 9 grid points)

| Sweep | Variant | T values | alpha values | Grid points |
|-------|---------|---------|-------------|-------------|
| `streaming_T` | streaming | {1, 2, 4} | best 3 from P1 | 9 |

**Total: 7 sweeps, 53 grid points, 53 runs (5 seeds batched per run).**

Each 200K-step run with a 4.3K-param network should take 1-3 min on GPU. Total wall time: ~1-3 hours sequential.


### Verification checklist

1. **Environment sanity:** Visualize a trajectory (scatter colored by time). Verify reflection at barriers/boundaries. Check reward regions produce r=1.
2. **V* sanity:** Plot V* heatmap. High values near rewards, smooth gradients in open areas, sharp drops across barriers.
3. **Single-run test:** 10K steps locally, no MLflow. MSVE should decrease. TD error should decrease.
4. **Linear baseline floor:** Linear model plateaus at high MSVE (can't represent barrier discontinuities). If nonlinear models don't beat it, barriers may need to be more impactful.
5. **Sweep sanity:** After Phase 1, at least one LR per variant shows decreasing MSVE. If all diverge, adjust LR range.
