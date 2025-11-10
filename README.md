# Feature Search Experiments

This repository provides tools for running experiments related to feature search and adaptive learning algorithms.

## Setup

Before using this repository, complete the following setup steps:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install the module in development mode:**
   ```bash
   pip install -e .
   ```

You may need to install JAX separately if the first command does not automatically install the CUDA-enabled version of JAX.

## Running Experiments

### Quick Start

To run your first experiment:

1. Navigate to the feature search directory:
   ```bash
   cd phd/feature_search
   ```

2. Run the example experiment:
   ```bash
   python scripts/jax_full_feature_search.py
   ```
   Or if you want to run it on CPU (much slower), then add the `device=cpu` flag:
   
   ```bash
   python scripts/jax_full_feature_search.py device=cpu
   ```

The `phd/feature_search/scripts/` directory contains many scripts that were used to run prior experiments, but as of now, I only use the `jax_full_feature_search.py`. This script uses a JAX rewrite of the codebase that was originally written in PyTorch, and it is significantly faster on GPU (~100x faster than PyTorch). With the right config, it can be used to replicate all of my major results.

When you run the `jax_full_feature_search.py` script, it defaults to using the config file located at `phd/feature_search/conf/full_feature_search.yaml`. This config specifies a stationary (`task.flip_rate=0.0`) 2-layer (`task.n_layers=2`) target network with 20 hidden units (`task.hidden_dim=20`) for the task, and a 2-layer (`model.n_layers=2`) network with 1024 hidden units (`model.hidden_dim=1024`) for the learner. The input layer of the learning network is frozen (`model.n_frozen_layers=1`), with feature search applied at a rate of 0.5% of the total number of hidden units per step (`feature_recycling.recycle_rate=0.005`). The output layer is trained using Autostep (`optimizer.name=idbd` + `optimizer.autostep=true`). This config further inherets from the `defaults.yaml` config file, which contains documentation on what each parameter does.

There are multiple ways you can customize the experiment:

<!-- ### Running Sweeps

To run parameter sweeps using Weights & Biases:

1. **Create or select a sweep configuration:**
   Example sweep configs can be found in `conf/wandb_sweeps/`

2. **Initialize the sweep from the feature_search directory:**
   ```bash
   cd phd/feature_search
   wandb sweep path_to_sweep_config.yaml
   ```

3. **Run the sweep agent:**
   After initializing, you'll receive a command in the format:
   ```bash
   wandb agent sweep_id
   ```
   
   Copy and run this command to start executing sweep runs. -->

### Custom Experiment Method 1: Command Line Overrides

Override configuration values directly from the command line. For example, we could train for 10k steps instead of the default 1M steps as follows:

```bash
python scripts/jax_full_feature_search.py train.total_steps=10000
```

### Custom Experiment Method 2: Modify Default Config

Edit the configuration file at `phd/feature_search/conf/full_feature_search.yaml`. For example, to use Adam instead of Autostep, you would change the optimizer name:

```yaml
optimizer:
  name: adam
```

### Custom Experiment Method 3: Create New Config Files

Create a new configuration file (e.g., `phd/feature_search/conf/autostep_cfg.yaml`):

```yaml
defaults:
  - defaults
  - _self_

train:
  optimizer: idbd

idbd:
  autostep: true
```

Then run with your custom config:

```bash
python scripts/experiment_template.py --config-name=autostep_cfg
```

### Custom Experiment Method 4: Create New Scripts

For experiments requiring functionality not supported by existing scripts, create new script files following the patterns in the existing examples.

## Available Scripts

| Script | Description |
|--------|-------------|
| `experiment_template.py` | Simple, extendable example for linear regression problems |
| `rupam_experiment.py` | Framework implementing experiments from [Mahmood & Sutton (2013)](http://incompleteideas.net/papers/MS-AAAIws-2013.pdf) |
| `full_feature_search.py` | Extended version of `rupam_experiment.py` with additional features like distractor hidden units and target noise |

## Analysis

### Downloading Experiment Data

Use the `wandb_download.py` script to download experiment results from Weights & Biases:

```bash
python phd/feature_search/analysis/wandb_download.py [options]
```

**Common usage patterns:**

```bash
# Download all data from a project
python wandb_download.py --project your_project_name

# Download data from specific sweeps
python wandb_download.py --project your_project_name --sweeps sweep_id1 sweep_id2
```

**Key options:**
- `--sweeps`: Download data from specific sweep IDs
- `--tags`: Download data from runs with specific tags
- `--data_type`: Choose `run` (default, for metrics) or `table` (for artifact data)
- `--history_vars`: Specify which metrics to download (default: all)
- `--include_crashed`: Include crashed runs in download

**Output:**
The script generates two CSV files:
- `{project}_config_data.csv`: Run configurations and metadata
- `{project}_{data_type}_data.csv`: Logged metrics and results

View existing analysis notebooks in the `analysis/` directory for examples of how to process and visualize the downloaded data.

## References

- Sutton, R. S. (1992). Adapting bias by gradient descent: An incremental version of delta-bar-delta. *Proceedings of the Tenth National Conference on Artificial Intelligence*.
- Mahmood, A. R., & Sutton, R. S. (2013). Representation search through generate and test. *Proceedings of the 2013 AAAI Spring Symposium*.

---

For more detailed documentation and advanced usage, please refer to the individual script files and configuration examples.