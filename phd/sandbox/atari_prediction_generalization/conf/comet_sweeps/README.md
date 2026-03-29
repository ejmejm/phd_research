# Return Prediction Sweeps

These sweeps train a simple ResNet to predict returns on a sequence of Atari games using supervised learning. The key experimental variable is `train.reinit_at_game_boundary`: when `true`, the network is reset between games; when `false`, it carries over. A network that carries over could perform worse (due to plasticity loss), better (due to generalization), or no differently.

Two optimizers are swept:

- **Adam** (`adam_sweep.yaml`): grid search over learning rate.
- **UPGD** (`upgd_sweep.yaml`): grid search over learning rate, beta_utility, sigma, and weight_decay.

Both sweeps run 5 seeds per configuration.

## Sweep Runs

### v1 (no return rescaling)

| Config | Sweep ID | Trials | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|----------|----------|
| adam_sweep | `050debd5683b45509e97e08ebf81ee42` | 20 | ~2.5 | 50 | 12h | 5 |
| upgd_sweep | `f9ee8fdda7fd4cedbe32a3dcf2b81656` | 600 | ~2.5 | 1500 | 12h | 150 |

Each 12h agent completes ~4 trials on average (reinit=True is slower due to JIT warm-up).

```bash
# adam_sweep
sbatch --array=1-5 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=2 --mem=16G --time=12:00:00 \
  launch_comet_agent.sbatch -s 050debd5683b45509e97e08ebf81ee42 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization

# upgd_sweep
sbatch --array=1-150%56 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=2 --mem=16G --time=12:00:00 \
  launch_comet_agent.sbatch -s f9ee8fdda7fd4cedbe32a3dcf2b81656 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization
```

### v2 (with return rescaling; upgd sigma=1.0 removed)

Ran on Vulcan (L40S). Measured resource usage:

| Resource | Measured | Recommended (with 30% overhead) |
|----------|----------|--------------------------------|
| Memory (peak) | ~9.2 GB avg, 10.5 GB max | `--mem=14G` |
| CPUs | ~1.25 effective | `--cpus-per-task=2` |
| Time per trial | ~56 min at 772 it/s | — |
| Trials per 12h job | ~10 | — |

To estimate num jobs: `ceil(num_trials / 10 * 1.3)`.

| Config | Sweep ID | Trials | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|----------|----------|
| adam_sweep_v2_rescaled | `d60d25541b0b458ca7f4f1b18bacff77` | 14 | ~0.93 | 13 | 12h | 5 |
| upgd_sweep_v2_rescaled | `a27e851d8cc74eaaae999a985bc9de63` | 480 | ~0.93 | 448 | 12h | 65 |

53 of 140 UPGD jobs did actual work; the rest found the sweep already completed. 65 jobs (ceil(480/10*1.3)) would have been sufficient.

```bash
# adam_sweep_v2_rescaled (Vulcan, L40S)
sbatch --array=1-5 --gpus-per-node=1 \
  --cpus-per-task=2 --mem=14G --time=12:00:00 \
  launch_comet_agent.sbatch -s d60d25541b0b458ca7f4f1b18bacff77 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization

# upgd_sweep_v2_rescaled (Vulcan, L40S)
sbatch --array=1-65%50 --gpus-per-node=1 \
  --cpus-per-task=2 --mem=14G --time=12:00:00 \
  launch_comet_agent.sbatch -s a27e851d8cc74eaaae999a985bc9de63 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization
```
