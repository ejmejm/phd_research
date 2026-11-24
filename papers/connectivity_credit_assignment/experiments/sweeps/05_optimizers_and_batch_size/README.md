# 05 — optimizers and batch sizes

Does Figure 1's gap between block-sparse and dense (24 hidden units each, 4-task
multi-MNIST) survive adaptive optimizers and minibatch SGD? Appendix D reports the results.

## Setup

Everything not listed is Figure 1's: `01_matched_and_scaling/base_*.yaml`.

- **Optimizers.** Adam (`optim.py`'s, betas 0.9/0.999, eps 1e-8) and RMSProp
  (optax, decay 0.99, eps 1e-8), batch size 1.
- **Batch sizes.** SGD with batch sizes 1, 2, 4, 8, 16, 32. The problem is held
  fixed in samples: 500K samples, a permutation every 800 samples, logging every
  800 and test evaluation every 8,000 (stationary). Each is divided by the batch
  size to get steps, so a batch of 32 gets 25 updates per permutation.

## Two passes

**Pass 1, `sweep/`.** A step-size grid per configuration, seeds 0–4, 500K
samples: Adam/RMSProp 2⁻¹⁷ to 2⁻⁷, batch sizes 2⁻¹² to 2⁻². The winner is the
best 5-seed mean over the last 10%, using test accuracy on the stationary
problem and online accuracy on the non-stationary one, as in `01`.

**Final optimizer runs, `best/`.** Non-stationary Adam and RMSProp use 30 seeds
(101–130), with metrics every 50 steps. The batch-size figure uses the five-seed
sweep runs and has no confidence bands. No 30-seed batch-size runs are required.

Analysis: `analysis/07_optimizers_batch_size.ipynb`. The final optimizer panels
compare SGD, Adam, and RMSProp separately for each network. Both adaptive
optimizers perform worse than SGD here. The batch-size figure shows that the
connectivity gap persists from batch size 1 to 32. The notebook also retains
the stationary controls.

## Sweep runs

About 40 s per batch-size-1 trial locally (~15K steps/s plus ~5 s startup); larger
batches take fewer steps, so startup dominates them.

| Config | Sweep ID | Trials | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|----------|----------|
| sweep/optimizers_stationary | `2cfcdf306c9b4f4ab1bcd62b5a040d5b` | 44 | 0.02 | 0.9 | 3h | 4 |
| sweep/optimizers_nonstationary | `d04d7368b3eb4a11baa35cd25b1ba6cf` | 44 | 0.02 | 0.9 | 3h | 4 |
| sweep/batch_size_stationary | `795074ac2db74df5b3e32cf58328974d` | 132 | 0.02 | 2.6 | 3h | 10 |
| sweep/batch_size_nonstationary | `6d7d750c06e7443b95ac1cc1ea615276` | 132 | 0.02 | 2.6 | 3h | 10 |

```bash
cd ~/jobs
P=~/scratch/phd_research/papers/connectivity_credit_assignment
A="--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=6G --time=03:00:00"
sbatch --array=1-4  $A launch_comet_agent.sbatch -s 2cfcdf306c9b4f4ab1bcd62b5a040d5b -p $P
sbatch --array=1-4  $A launch_comet_agent.sbatch -s d04d7368b3eb4a11baa35cd25b1ba6cf -p $P
sbatch --array=1-10 $A launch_comet_agent.sbatch -s 795074ac2db74df5b3e32cf58328974d -p $P
sbatch --array=1-10 $A launch_comet_agent.sbatch -s 6d7d750c06e7443b95ac1cc1ea615276 -p $P
```

## Best runs

| Config | Sweep ID | Trials | Cluster | Job time | Num jobs |
|--------|----------|--------|---------|----------|----------|
| best/optimizers_nonstationary_log50 | `828badee12ab4a9183fe3c7dc2a73734` | 4 | Vulcan (L40S) | 1h | 4 |

About 4 min per trial at ~3,900 steps/s. Agents that finished early reran the
Adam block-sparse trial three more times with identical results, so the analysis
keeps one run per cell.

```bash
# Vulcan, job 2314954
sbatch --array=1-4 --gpus-per-node=1 --cpus-per-task=1 --mem=6G --time=01:00:00 \
  launch_comet_agent.sbatch -s 828badee12ab4a9183fe3c7dc2a73734 \
  -p $HOME/scratch/phd_research/papers/connectivity_credit_assignment
```
