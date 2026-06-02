# set_analysis_v2 — Comet sweeps for Compute Canada

Comet ports of the `weight_pruning_scaling/00_experimental/set/sweep_2/` mlflow
sweeps, in both a **stationary** and a **non-stationary** variant. 8 tasks; all
four sit at the 25,408-weight block-sparse budget (32 units at n=8) so SET,
random_sparse and the existing block-sparse results compare at equal total
connection count. Each trial vmaps 5 seeds (0-4); the `command` still points
at the base Hydra configs under `weight_pruning_scaling/` (resolved at runtime
relative to `experiments/`).

Layout mirrors `weight_pruning_scaling`: split by regime, then `sweep/`.

```
01_stationary/sweep/{random_sparse,set}.yaml
02_nonstationary/sweep/{random_sparse,set}.yaml
```

Project: `set-analysis-sweep-v2`.

**Stationary** — `permute_period=0`, test eval every 25k steps, objective
`asymptotic_test_loss`.
**Non-stationary** — `permute_period=250` (each event re-permutes one random
task -> 250*8 = ~2000 steps/task), `eval_freq=0` (no test split), objective
`asymptotic_loss`.

- **random_sparse** — fixed 183 units at the 2-task winning sparsity
  ([1,256]/[1,20]), sweep LR only. 5 LR = **5 trials**.
- **set** — `target_hidden_units {96,192,384}` x `init_mode {epsilon,uniform_p}`
  x `prune_metric {magnitude,utility}` x `zeta {0.01,0.05,0.1,0.2,0.3}` x 5 LR
  = **300 trials**.

## Sweep Runs

Throughput measured at ~2200 steps/s on a Nibi H100 MIG slice (1g.10gb, 5-seed
vmap; the 4080 hits ~6000 but the small-batch workload runs slower on the MIG).
Per trial = 500k steps (~227s) + ~12s compile + (stationary only) ~20 evals x
~5s = ~0.09 hr/trial stationary, ~0.07 hr/trial non-stationary. Totals include
a 1.4x buffer.

| Config | Sweep ID | Trials | Steps/sec (MIG) | Hrs/trial | Total hrs (x1.4) | Job time | Num jobs |
|--------|----------|--------|-----------------|-----------|------------------|----------|----------|
| 01_stationary/sweep/random_sparse | `3c7b3b2be79a467bad339616f0a00017` | 5 | ~2200 | ~0.09 | 0.6 | 3h | 1 |
| 01_stationary/sweep/set | `115211a82b564e139f6862d99b6c3372` | 300 | ~2200 | ~0.09 | 39 | 6h | 6 |
| 02_nonstationary/sweep/random_sparse | `e0ff9ee6e4474edba051890381c5b07a` | 5 | ~2200 | ~0.07 | 0.5 | 3h | 1 |
| 02_nonstationary/sweep/set | `f1163eecb71c4a4591ed85583eb4f0e2` | 300 | ~2200 | ~0.07 | 29 | 6h | 5 |

Total: 13 jobs across the four sweeps (all <50, within rate limits). Each
worker pulls trials continuously until the sweep is exhausted or the job ends,
so the arrays finish in ~4-5 h wall-clock.

```bash
# --- Test jobs (1h, 1 worker each) — run first to confirm the pipeline ---
# stationary_set test (train_set.py path)
sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=8G --time=01:00:00 \
  launch_comet_agent.sbatch -s 115211a82b564e139f6862d99b6c3372 \
  -p $HOME/scratch/phd_research/phd/structure_search

# stationary_random_sparse test (train_weight_pruning.py path)
sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=8G --time=01:00:00 \
  launch_comet_agent.sbatch -s 3c7b3b2be79a467bad339616f0a00017 \
  -p $HOME/scratch/phd_research/phd/structure_search

# --- Full arrays ---
# 01_stationary/sweep/random_sparse (5 trials)
sbatch --array=1-1 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=8G --time=03:00:00 \
  launch_comet_agent.sbatch -s 3c7b3b2be79a467bad339616f0a00017 \
  -p $HOME/scratch/phd_research/phd/structure_search

# 01_stationary/sweep/set (300 trials)
sbatch --array=1-6 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=8G --time=06:00:00 \
  launch_comet_agent.sbatch -s 115211a82b564e139f6862d99b6c3372 \
  -p $HOME/scratch/phd_research/phd/structure_search

# 02_nonstationary/sweep/random_sparse (5 trials)
sbatch --array=1-1 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=8G --time=03:00:00 \
  launch_comet_agent.sbatch -s e0ff9ee6e4474edba051890381c5b07a \
  -p $HOME/scratch/phd_research/phd/structure_search

# 02_nonstationary/sweep/set (300 trials)
sbatch --array=1-5 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=8G --time=06:00:00 \
  launch_comet_agent.sbatch -s f1163eecb71c4a4591ed85583eb4f0e2 \
  -p $HOME/scratch/phd_research/phd/structure_search
```

<!-- sbatch commands added once sweep IDs are created and throughput measured. -->
