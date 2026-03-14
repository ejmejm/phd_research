# Return Prediction Sweeps

These sweeps train a simple ResNet to predict returns on a sequence of Atari games using supervised learning. The key experimental variable is `train.reinit_at_game_boundary`: when `true`, the network is reset between games; when `false`, it carries over. A network that carries over could perform worse (due to plasticity loss), better (due to generalization), or no differently.

Two optimizers are swept:

- **Adam** (`adam_sweep.yaml`): grid search over learning rate.
- **UPGD** (`upgd_sweep.yaml`): grid search over learning rate, beta_utility, sigma, and weight_decay.

Both sweeps run 5 seeds per configuration.

## Sweep Runs

| Config | Sweep ID | Trials | Steps/sec | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|-----------|----------|----------|
| adam_sweep | `8423804ddfea419ba4493368f956a131` | 20 | 450 | 1.6 | 32 | 12h | 4 |
| upgd_sweep | `c7e7880df0ed45c58c8788a67163d260` | 600 | 450 | 1.6 | 960 | 12h | 112 |

```bash
# adam_sweep
sbatch --array=1-4 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=8G --time=12:00:00 \
  launch_comet_agent.sbatch -s 8423804ddfea419ba4493368f956a131 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization

# upgd_sweep
sbatch --array=1-112%56 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=8G --time=12:00:00 \
  launch_comet_agent.sbatch -s c7e7880df0ed45c58c8788a67163d260 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization
```
