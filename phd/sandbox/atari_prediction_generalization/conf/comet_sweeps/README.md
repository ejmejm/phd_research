# Return Prediction Sweeps

These sweeps train a simple ResNet to predict returns on a sequence of Atari games using supervised learning. The key experimental variable is `train.reinit_at_game_boundary`: when `true`, the network is reset between games; when `false`, it carries over. A network that carries over could perform worse (due to plasticity loss), better (due to generalization), or no differently.

Two optimizers are swept:

- **Adam** (`adam_sweep.yaml`): grid search over learning rate.
- **UPGD** (`upgd_sweep.yaml`): grid search over learning rate, beta_utility, sigma, and weight_decay.

Both sweeps run 5 seeds per configuration.

## Sweep Runs

| Config | Sweep ID | Trials | Steps/sec | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|-----------|----------|----------|
| adam_sweep | `0dead6cd7bc24556a0bc7f04d4e80a47` | 20 | 450 | 1.6 | 32 | 12h | 4 |
| upgd_sweep | `1d8e4dcf080643aea40769e2c4a24b53` | 600 | 450 | 1.6 | 960 | 12h | 113 |

```bash
# adam_sweep
sbatch --array=1-3 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=6G --time=12:00:00 \
  launch_comet_agent.sbatch -s 0dead6cd7bc24556a0bc7f04d4e80a47 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization
# (+ 1 individual job already submitted: 10150872)

# upgd_sweep
sbatch --array=1-112%56 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=6G --time=12:00:00 \
  launch_comet_agent.sbatch -s 1d8e4dcf080643aea40769e2c4a24b53 \
  -p $HOME/scratch/phd_research/phd/sandbox/atari_prediction_generalization
# (+ 1 individual job already submitted: 10150873)
```
