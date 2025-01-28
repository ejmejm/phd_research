# Model Scaling Experiments

This codebase implements experiments to investigate the relationship between model size and performance. The core question being investigated is: "Is a larger model always better?"

## Experiment Design

The experiments train fully-connected networks on two standard image classification tasks:
- MNIST (28x28 grayscale digits)
- CIFAR-10 (32x32x3 color images)

Three key architectural parameters are varied:
1. **Network Width**: The size of hidden layers [32, 64, 128, 256, 512]
2. **Network Depth**: Number of hidden layers [2, 3, 4, 5, 6] 
3. **Sparsity**: Fraction of weights randomly masked to zero at initialization [0%, 25%, 50%, 75%, 90%]

Each configuration is run with 5 different random seeds for statistical significance.

## Training Details

- Models are trained for a fixed number of total training samples (3M by default) rather than epochs
- Training progress is logged every 100 steps
- Validation is performed every 500 steps
- RMSprop optimizer is used by default

## Metrics Tracked

During training:
- Training loss and accuracy
- Validation loss and accuracy 
- Number of steps taken
- Number of samples seen
- Current epoch (can be fractional)
- Number of active (non-masked) parameters

Summary metrics:
- Early performance (avg over first 10% of training)
- Final performance (avg over last 10% of training)
- Total steps and samples seen
- Final epoch reached

## Running Experiments

Single run:
```bash
python run_experiment.py
```

Full sweep:
```bash
wandb sweep conf/sweeps/scaling_sweep.yaml
```

The sweep will test all combinations of width, depth and sparsity across both datasets and 5 seeds, resulting in 1250 total runs (5×5×5×5×2).

## Configuration

Default settings are in `conf/defaults.yaml`. The sweep configuration is in `conf/sweeps/scaling_sweep.yaml`.

Key parameters can be overridden via command line, e.g.:
```bash
python run_experiment.py model.width=256 model.depth=4 data.dataset=cifar10
```