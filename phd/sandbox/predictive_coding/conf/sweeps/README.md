# Streaming iPC Sweep Configs

All sweeps use `experiment: streaming-ipc-sweeps` and share a common MLflow experiment.

## Sweep Overview

| Phase | Directory | Sweeps | Grid points | Description |
|-------|-----------|--------|-------------|-------------|
| 1 | `variant_lr_sweep/` | 5 | 20 | LR tuning per variant (streaming, forward_init, ema, BP streaming, BP shuffled) |
| 2 | `T_gamma_sweep/` | 2 | 28 | Inference steps (T) and inference LR (gamma) for streaming iPC |
| 3 | `ema_beta_sweep/` | 1 | 16 | EMA blending parameter (beta) |
| 4 | `rotation_speed/` | 2 | 24 | Rotation speed (delta_degrees) for streaming vs forward_init |
| 5 | `architecture/` | 2 | 20 | Network depth and width |

**Total**: 12 sweeps, 108 grid points, ~324 runs (3 seeds each)

## Running

All sweeps are independent and can run in parallel:

```bash
cd phd/sandbox/predictive_coding
mlflow-sweep --config conf/sweeps/variant_lr_sweep/streaming_ipc_lr.yaml
```

## Analysis

After sweeps complete, download data and run the analysis notebook:

```bash
mlflow_download --tracking-uri=$MLFLOW_TRACKING_URI --experiment=streaming-ipc-sweeps --output-dir=analysis/sweep_comparison/data/
```
