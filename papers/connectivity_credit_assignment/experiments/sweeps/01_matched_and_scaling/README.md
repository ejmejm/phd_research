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
