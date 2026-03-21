# Streaming Reinforcement Learning Sandbox

Sandbox for experimenting with the streaming RL algorithms (StreamAC, StreamQ) and ObGD
optimizer from [Elsayed et al. (2024), "Streaming Deep Reinforcement Learning Finally
Works"](https://arxiv.org/abs/2410.14606). This is a PyTorch reimplementation of their
approach for local experimentation.

## Motivation

Standard deep RL algorithms (PPO, SAC, DQN) rely on experience replay and batch updates.
The streaming RL work by Elsayed et al. shows that fully online, per-step updates with
eligibility traces can work at scale when combined with ObGD's adaptive step-size scaling
and sparse initialization. This sandbox reimplements their algorithms to test them on
continuous control (MuJoCo) and discrete action (Atari) domains, and to evaluate the
quality of features learned by streaming value networks.

## Components

### Core

- **`core/obgd.py`** — ObGD (Observed-Gradient Descent) optimizer from Elsayed et al.
  (2024). PyTorch implementation with eligibility traces (gamma, lambda), adaptive
  step-size scaling (kappa), and trace reset support for episode boundaries.
- **`core/layers.py`** — Custom `LayerNormalization` wrapper around PyTorch's functional
  layer norm.
- **`core/param_init.py`** — Sparse weight initialization: randomly zeros out a fraction of
  weights per output neuron while using fan-in scaled uniform or normal init for the rest.
- **`core/processing.py`** — Gymnasium environment wrappers: `NormalizeObservation` (running
  mean/std), `ScaleReward` (discounted reward normalization), `AddTimeInfo` (appends
  normalized timestep), and `linear_schedule` for epsilon decay.

### Algorithms

- **`core/algorithms/stream_ac.py`** — StreamAC (Streaming Actor-Critic) for continuous
  action spaces. Gaussian policy with separate actor/critic networks, 2 hidden layers
  (128 units), LeakyReLU + LayerNorm, 90% sparse init, trained with ObGD.
- **`core/algorithms/stream_q.py`** — StreamQ (Streaming Q-Learning) for discrete action
  spaces. CNN architecture (3 conv layers + 2 FC), epsilon-greedy with linear decay,
  trained with ObGD.

### Notebooks

- **`notebooks/ant_feature_eval.ipynb`** — Trains multiple StreamAC agents on Ant-v4 (100K
  steps, 3 seeds), saves value network checkpoints every 10K steps, then evaluates the
  learned features by freezing all layers except a final linear head that pools across 30
  parallel value functions. Tracks MAE of TD-errors and visualizes feature weight magnitudes.
- **`notebooks/test.ipynb`** — Trains StreamQ on PongNoFrameskip-v4 (300K steps) with
  standard Atari preprocessing (frame stacking, grayscale, 84x84 resize). Supports
  checkpoint saving/loading and GIF rendering of agent behavior.
