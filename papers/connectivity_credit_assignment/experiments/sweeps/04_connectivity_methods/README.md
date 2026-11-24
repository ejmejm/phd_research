# Connectivity methods

Figure 4 compares dense, block-sparse, static sparse, SET, DEEP-R, and
Continual SET on non-stationary multi-MNIST. There are 16 subproblems,
one label permutation every 125 steps, and approximately 203,264 connections
per network. The dense network uses 16 hidden units to match the budget.

All methods use `padded_mlp`. Each method has its own initialization and
update rule, so the comparison does not isolate the utility criterion.
The static sparse baseline uses SET's initialization without rewiring.

## Baseline configurations and provenance

The original baseline sweeps remain in `sweep/`. They use 5 seeds per
configuration and 800,000 steps. `set_topup.yaml` and `deep_r_topup.yaml`
extend the original grids; `set_v3.yaml` and `deep_r_v3.yaml` hold the
selected step-size fixed and explore the remaining parameters.
Notebook 04 retains these sensitivity analyses.

The `best/` directory retains the original 800,000-step, 30-seed baseline
configurations. The current manuscript uses the corresponding `best_2m/`
configurations. Seeds 101–130 are split into two batches of 15 for GPU memory.
Deduplicate by configuration and seed offset before combining the batches.

The final baseline configurations are:

- Dense: 16 hidden units, step-size 2^-10.
- Block-sparse: 256 hidden units, step-size 2^-8.
- Static sparse: 768 hidden units, step-size 2^-7.
- SET: 256 hidden units, step-size 2^-8, rewiring every 1000 steps, fraction 0.005.
- DEEP-R: 768 hidden units, step-size 2^-7, L1 coefficient 1e-4, noise ratio 3.

SET initializes with an Erdős–Rényi mask and Glorot weights, then regrows
zero-valued weights at random positions across each layer. Its replacement
fraction is derived from a rate of 5e-6 per sample. DEEP-R uses its own
uniform mask and normal weight initialization, with output density ratio 10.
The base configuration records these choices.

Algorithm event periods must divide `train.log_freq`. A diverged trial
remains part of the sweep record, but is excluded from the best-cell score.
Initial pilots remain in `sweep/pilot_*.yaml` as records of the baseline checks.

## Continual SET

The current method tracks a running average of each connection's absolute
contribution. Every 500 steps, it removes the lowest-utility 1% of connections
in each layer. It adds the same number at uniformly selected inactive positions
within that layer. New connections receive the median utility of the surviving
connections. Initial and replacement weights use LeCun uniform at the receiving
unit's sparse input count. Layer counts stay fixed; input counts per unit can change.

### Supporting sweep

`sweep/set_utility_median.yaml` records the sweep used to choose the parameters.
It uses 5 seeds and 800,000 steps per configuration, with these values:

- Hidden units: 128, 256, 512.
- Step-size: 2^-9, 2^-8, 2^-7, 2^-6.
- Replacement fraction: 0.01, 0.003, 0.001, 0.0003, 0.0001.
- Utility decay: 0.99, 0.999, 0.9999.

The selected values are 256 hidden units, step-size 2^-8, fraction 0.01,
and decay 0.9999. Sweep ID: `06b6c13b54b648c188a8c907233aa038`.
This sweep used `row_lecun`, which preserves each unit's input count.
The final Figure 4 run reuses those values with `layer_lecun`.
Keep the row-based implementation to reproduce this supporting sweep.

### Figure 4 runs

`best_2m/` contains the six configurations used in Figure 4. All use 2 million
steps and seeds 101–130, split into two batches of 15 vmapped seeds.
The project is `paper-weight-pruning-connectivity-best`.

- Dense: `70f8b91063574835a53626fcb11145c2`.
- Block-sparse: `1801f550941a4e7ba82a3a05309b37f0`.
- Static sparse: `8e5540bfb6f54543ace713bf3d232173`.
- SET: `1abf1193d3104e418dc5d6773bb76516`.
- DEEP-R: `c1841f7c388041dca4b3e72809394d69`.
- Continual SET: `41b028abe91c433986d92427150b17e2`.

The Continual SET configuration is `best_2m/continual_set_median_global.yaml`.
Its sweep name is `04_continual_set_median_global_n16_best_2m`.
Vulcan job array `2389004` completed both batches, each in about 56 minutes.
The source snapshot was
`/home/ejmejm/scratch/experiment_snapshots/continual_set_median_global_41b028ab_v2`.
Its archive SHA256 was `ede3c3b39c5d34370929546429a27321d40a523870817bf80685e9e940915b34`.

Use these two distinct seed batches:

- Seeds 101–115: `e2ba8c13082b422e8f6ee78f74b15cf7`.
- Seeds 116–130: `3db8a8b2cd504af6b46a72629707c959`.

Exclude `f02257c24d6f4270989221b2b48204b1`, which repeats seeds 101–115.
Notebook 04 removes duplicate seed batches before it plots either measure.
Final-10% accuracy is 78.1665%. Final pooled path purity is 0.0953575.
The run closes about 87% of the accuracy gap from dense to block-sparse.

The earlier utility variants and their plot drafts were removed after revision 2.
The original baseline sweeps and 800,000-step baseline configurations remain
as provenance for the original manuscript.
