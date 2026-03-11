# Return Prediction Sweeps

These sweeps train a simple ResNet to predict returns on a sequence of Atari games using supervised learning. The key experimental variable is `train.reinit_at_game_boundary`: when `true`, the network is reset between games; when `false`, it carries over. A network that carries over could perform worse (due to plasticity loss), better (due to generalization), or no differently.

Two optimizers are swept:

- **Adam** (`adam_sweep.yaml`): grid search over learning rate.
- **UPGD** (`upgd_sweep.yaml`): grid search over learning rate, beta_utility, sigma, and weight_decay.

Both sweeps run 5 seeds per configuration.

## Sweep Runs

| Config | Sweep ID | Trials | Steps/sec | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|-----------|----------|----------|
| dummy_test | `975f3206e29f42149c06f4da5ed92484` | 1 | 450 | 1.6 | 1.6 | 30m | 1 |
| adam_sweep | `f243b2af91454b759d300c702af397c7` | 20 | 450 | 1.6 | 32 | 12h | 4 |
| upgd_sweep | `204ca4dc371945d1b1e9a45418ef67a9` | 600 | 450 | 1.6 | 960 | 12h | 112 |

```bash
# dummy_test
sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=6G --time=00:30:00 \
  launch_comet_agent.sbatch -s 975f3206e29f42149c06f4da5ed92484 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization

# adam_sweep
sbatch --array=1-4 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=6G --time=12:00:00 \
  launch_comet_agent.sbatch -s f243b2af91454b759d300c702af397c7 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization

# upgd_sweep
sbatch --array=1-112%50 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=6G --time=12:00:00 \
  launch_comet_agent.sbatch -s 204ca4dc371945d1b1e9a45418ef67a9 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization
```
