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

## Running Experiments

### Quick Start

To run your first experiment:

1. Navigate to the feature search directory:
   ```bash
   cd phd/feature_search
   ```

2. Run the example experiment:
   ```bash
   python scripts/experiment_template.py
   ```

The `experiment_template` script demonstrates a simple regression problem where a single-layer prediction network learns to predict the output of a single-layer target network with 10 distractors in the input.

### Running Sweeps

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
   
   Copy and run this command to start executing sweep runs.

## Creating Custom Experiments

You can customize experiments in several ways:

### Method 1: Command Line Overrides

Override configuration values directly from the command line:

```bash
python scripts/experiment_template.py train.optimizer=idbd idbd.autostep=true
```

### Method 2: Modify Default Config

Edit the default configuration file at `conf/defaults.yaml`. For example, to use Autostep instead of Adam:

```yaml
train:
  optimizer: idbd

idbd:
  autostep: true
```

### Method 3: Create New Config Files

Create a new configuration file (e.g., `conf/autostep_cfg.yaml`):

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

### Method 4: Create New Scripts

For experiments requiring functionality not supported by existing scripts, create new script files following the patterns in the existing examples.

## Available Scripts

| Script | Description |
|--------|-------------|
| `experiment_template.py` | Simple, extendable example for linear regression problems |
| `rupam_experiment.py` | Framework implementing experiments from [Mahmood & Sutton (2013)](http://incompleteideas.net/papers/MS-AAAIws-2013.pdf) |
| `full_backbone_recycling.py` | Extended version of `rupam_experiment.py` with additional features like distractor hidden units and target noise |

## Available Tasks

Tasks are defined in `phd/feature_search/core/tasks.py` and include:

### `dummy`
Generates random inputs and outputs for testing purposes.

### `linear_geoff`
Implements the bit-flipping regression task from the [original IDBD paper](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf). Features include:
- Some features are distractors, others have ideal weights of -1 or +1
- One useful feature weight flips every 20 steps
- Regression problem structure

### `static_linear_geoff`
Similar to `linear_geoff`, but with static ideal weights (no bit-flipping throughout the task).

### `nonlinear_geoff`
A generalized version of the Geoff task supporting:
- Multi-layer target networks
- Hidden layer activations
- Time-varying target network weights
- Binary or Kaiming uniform weight initialization

This task is demonstrated in [this talk](https://youtu.be/qcdNaVAyeQ4?si=Tc2xkqTTvVJnKu_S) and detailed in [Mahmood & Sutton (2013)](http://incompleteideas.net/papers/MS-AAAIws-2013.pdf).

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