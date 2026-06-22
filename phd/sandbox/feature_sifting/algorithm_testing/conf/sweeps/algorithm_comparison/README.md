# Algorithm comparison: LMS vs. CBP+Autostep on feature sifting

Step-size / hyperparameter sweeps for each algorithm under a **stationary**
(`01_stationary`, `task.flip_period=-1`) and a **non-stationary**
(`02_nonstationary`, `task.flip_period=100`) feature-sifting task, followed by
30-seed runs of the best configuration found in each sweep.

## Layout

```
algorithm_comparison/
  base_cbp_autostep.yaml      # self-contained Hydra base (method=cbp_autostep)
  base_lms.yaml               # self-contained Hydra base (method=lms)
  01_stationary/              # flip_period=-1
    sweep/{cbp_autostep,lms}.yaml   # 5-seed hyperparameter sweeps
    best/{cbp_autostep,lms}.yaml    # 30-seed runs of the sweep winner
  02_nonstationary/           # flip_period=100
    sweep/{cbp_autostep,lms}.yaml
    best/{cbp_autostep,lms}.yaml
  03_convergence/             # long-horizon noise-floor convergence study
    cbp_replace_rate.yaml     # 50-seed, 10M-step replace_rate grid (2^-16 .. 2^-10)
```

`03_convergence/cbp_replace_rate.yaml` sweeps the pruning rate `replace_rate` over
`2^-16 .. 2^-10` on the stationary task (other CBP hyperparameters fixed at the 01
stationary best), 50 seeds × 10M steps each, to test whether CBP's noise floor
(`best_possible_noise`) converges to a positive, rate-dependent value or keeps
decreasing toward 0. Logged to the `feature_sifting_convergence` MLflow experiment.

Seeds are vmapped inside `run_experiment.py`, so each sweep trial / best config
is a single run over the whole seed set (not one run per seed).

## Replication

Run from the `phd/sandbox/feature_sifting/algorithm_testing/` directory.

### 1. Sweeps (5 seeds each)

```bash
# Stationary
mlflow-sweep conf/sweeps/algorithm_comparison/01_stationary/sweep/cbp_autostep.yaml
mlflow-sweep conf/sweeps/algorithm_comparison/01_stationary/sweep/lms.yaml
# Non-stationary
mlflow-sweep conf/sweeps/algorithm_comparison/02_nonstationary/sweep/cbp_autostep.yaml
mlflow-sweep conf/sweeps/algorithm_comparison/02_nonstationary/sweep/lms.yaml
```

`lms` is a learning-rate grid; `cbp_autostep` is a one-at-a-time sensitivity
sweep over `learning_rate`, `meta_learning_rate`, and `replace_rate` off a baseline.

### 2. Best-config follow-ups (30 seeds each)

After identifying the best config per sweep, run the fixed-config follow-ups:

```bash
# Stationary
mlflow-sweep conf/sweeps/algorithm_comparison/01_stationary/best/cbp_autostep.yaml
mlflow-sweep conf/sweeps/algorithm_comparison/01_stationary/best/lms.yaml
# Non-stationary
mlflow-sweep conf/sweeps/algorithm_comparison/02_nonstationary/best/cbp_autostep.yaml
mlflow-sweep conf/sweeps/algorithm_comparison/02_nonstationary/best/lms.yaml
```

### Best configs (winners of each sweep)

| Scenario | Method | Best hyperparameters |
|----------|--------|----------------------|
| Stationary | cbp_autostep | lr=2⁻², meta_lr=2⁻⁷, replace_rate=1e-4 |
| Stationary | lms | lr=2⁻¹³ |
| Non-stationary | cbp_autostep | lr=2⁻², meta_lr=2⁻⁷, replace_rate=1e-4 |
| Non-stationary | lms | lr=2⁻¹³ |

For the cbp_autostep sensitivity sweeps the baseline topped the ranking and every
one-axis deviation was worse, so the best config equals the sweep baseline.
