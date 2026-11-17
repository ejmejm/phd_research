# 01 — matched connectivity, and scaling the dense network

**Figures 1 and 2.** Stationary and non-stationary 4-task multi-MNIST.

Block-sparse at 24 units (19,056 weights) against dense at seven budgets,
6 to 384 units — 1x to 64x of that same base. Figure 1 compares the two at the
matched 1x budget; Figure 2 asks whether the dense network can buy its way out
with more parameters.

It can, in the stationary problem, at about 4x. It cannot in the
non-stationary one, where past 152,000 parameters it gets *worse*.

`sweep/` picks a step-size per cell over 5 seeds. `best/` re-runs the winners
over 30 seeds; those are the runs the figures are drawn from.

Analysis: `analysis/01_matched_and_scaling.ipynb`

## `sweep/dense_8x_permute_period.yaml` — rebuttal: permutation rate

Block-sparse (24 units) against 8x dense (48 units) as the permutation period
varies from 200 to 6400 steps. The block-sparse side and dense's 800-step cell
come from the pre-release sweeps `04_block_sparse_permute_period_lr` and
`02_dense_scaling_4task_nonstationary_lr`; this adds dense at the other five
periods, same seeds (0–4), step-sizes 2⁻¹¹ to 2⁻⁸. 20 trials.

With `log_freq` 1000, permutation events inside a 1000-step window are all
applied at its start, so periods 200 and 400 arrive as bursts of 5 and 2–3.

Sweep `84ef212f1e4944ae9d525566bdefece4`, ~45 s per trial, run on Nibi as:

```bash
sbatch --array=1-4 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 \
  --mem=6G --time=03:00:00 launch_comet_agent.sbatch \
  -s 84ef212f1e4944ae9d525566bdefece4 -p ~/scratch/phd_research/papers/connectivity_credit_assignment
```

Analysis: `analysis/05_dense_width_controls.ipynb`

## `sweep/dense_permute_period.yaml` — rebuttal: permutation rate at matched width

The same comparison with dense at block-sparse's 24 hidden units, matching
Figure 1. Dense-24's 800-step cell is in `02_dense_scaling_4task_nonstationary_lr`;
this adds the other five periods, seeds 0–4, step-sizes 2⁻¹² to 2⁻⁸. 25 trials.

Sweep `7c25804d241044d09652e51e41bac238`, ~35 s per trial on the local RTX 4080,
run locally with three agents:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m phd.research_utils.scripts.comet_sweep \
  -s 7c25804d241044d09652e51e41bac238   # x3, from the project root
```

Analysis: `analysis/06_matched_width_and_permutations.ipynb`
