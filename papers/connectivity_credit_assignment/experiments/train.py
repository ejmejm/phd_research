"""The single entry point for every experiment in this repository.

A run is a choice of model, connectivity algorithm, problem, and optimizer::

    python experiments/train.py \\
        model.type=padded_mlp model.init_strategy=block_sparse \\
        algorithm.name=static \\
        task.n_tasks=4 task.permute_period=800 \\
        optimizer.learning_rate=$(python -c 'print(2**-10)')

The configurations behind each figure live in ``experiments/sweeps/``; see the
README there for the exact commands.
"""

import hydra
from omegaconf import DictConfig

from multi_mnist.experiment import run_config
from multi_mnist.logging import finish_experiment, init_experiment


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg: DictConfig) -> None:
    cfg = init_experiment(cfg.project, cfg)
    try:
        run_config(cfg)
    finally:
        finish_experiment(cfg)


if __name__ == '__main__':
    main()
