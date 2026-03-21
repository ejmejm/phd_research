# StreamAC Scaling Experiments

Investigates how model size and the kappa hyperparameter affect the performance of
StreamAC (Streaming Actor-Critic) from [Elsayed et al. (2024), "Streaming Deep
Reinforcement Learning Finally Works"](https://arxiv.org/abs/2410.14606). JAX
reimplementation with multi-seed parallelism via `jax.vmap`.

## Motivation

StreamAC is a fully online, per-step RL algorithm that uses eligibility traces and the
ObGD optimizer (Elsayed et al. 2024). The key question here is: how does model
capacity (hidden dimension) interact with the kappa parameter (which controls adaptive
step-size scaling to prevent divergence) in this streaming setting? Does scaling up the
network help, hurt, or require careful kappa tuning?

## Components

### Core

- **`core/models.py`** — StreamACNet: actor/critic network with sparse initialization
  (90% sparsity), layer normalization, and configurable activations. Supports both
  discrete (softmax) and continuous (Gaussian) action spaces.
- **`core/optimizers.py`** — ObGD (Observed-Gradient Descent): online SGD with
  exponential eligibility traces (gamma, lambda) and kappa-based adaptive step-size
  scaling. Traces reset at episode boundaries.
- **`core/normalizers.py`** — Online observation normalization (Welford running
  mean/var) and reward scaling (discounted return variance).
- **`core/envs.py`** — Environment interface wrapping gymnax. Supports CartPole,
  MinAtar (Breakout, Freeway), Craftax, and Pendulum.

### Training

- **`scripts/stream_ac.py`** — Main training loop. Uses `jax.lax.scan` for efficient
  step sequences and `jax.vmap` to run all seeds in parallel without Python loops.
  Logs metrics (episode return, TD error, reward) to MLflow.

### Configuration

- **`conf/stream_ac.yaml`** — Base config: 1M steps, hidden_dim=64, 2 layers,
  leaky_relu, lr=1.0, gamma=0.99, lambda=0.8, kappa=3.0, entropy_coeff=0.01.
- **`conf/sweeps/model_size_kappa.yaml`** — Grid sweep: 5 hidden dims
  [32, 64, 128, 256, 512] x 6 kappa values [0.5, 1, 2, 4, 8, 16], 11 seeds each.

## Running

Single run:
```bash
python scripts/stream_ac.py
```

Sweep (via mlflow-sweeper):
```bash
mlflow-sweeper conf/sweeps/model_size_kappa.yaml
```
