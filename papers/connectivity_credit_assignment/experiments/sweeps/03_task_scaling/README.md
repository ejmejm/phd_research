# 03 — scaling the problem and the network

**Figure 4.** Block-sparse and dense from 4 to 512 hidden units, on problems
of 2 to 32 tasks, 5 seeds per cell.

`permute_period = 2000 / n_tasks`, so each individual task is re-permuted once
every 2000 steps on average no matter how many tasks there are — the amount of
non-stationarity per task is held fixed while the problem grows.

Measures how the cost of bad connectivity scales. At 32 tasks block-sparse
reaches about 86% while no dense network of any size passes about 31%.

One sweep per method, no `best/` stage: the sweep is the figure, with each
cell reported at its own best step-size. Block-sparse cells with fewer hidden
units than tasks fail fast by design.

Analysis: `analysis/03_task_scaling.ipynb`
