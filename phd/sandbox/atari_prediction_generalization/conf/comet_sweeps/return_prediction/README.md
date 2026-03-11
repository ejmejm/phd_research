# Return Prediction Sweeps

These sweeps train a simple ResNet to predict returns on a sequence of Atari games using supervised learning. The key experimental variable is `train.reinit_at_game_boundary`: when `true`, the network is reset between games; when `false`, it carries over. A network that carries over could perform worse (due to plasticity loss), better (due to generalization), or no differently.

Two optimizers are swept:

- **Adam** (`adam_sweep.yaml`): grid search over learning rate.
- **UPGD** (`upgd_sweep.yaml`): grid search over learning rate, beta_utility, sigma, and weight_decay.

Both sweeps run 5 seeds per configuration.
