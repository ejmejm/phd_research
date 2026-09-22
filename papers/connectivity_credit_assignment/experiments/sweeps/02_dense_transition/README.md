# 02 — the dense transition

**Figure 3.** 16-task non-stationary multi-MNIST, 256 hidden units, one task
re-permuted every 125 steps, 30 seeds.

Train block-sparse until it stabilizes, then flip the masks to all-ones at
step 250,000 so the cross-task weights become trainable. They are already
exactly zero, so the flip preserves the function and changes only which
weights gradients reach.

This rules out the alternative explanation for Figure 1: that dense networks
merely fail to *find* the block-sparse solution. Handed that solution, they
lose it.

One sweep, no `best/` stage — the method axis (`algorithm.transition_step`)
and the step-size axis sweep together, because the step-size each method
tolerates is itself part of the result.

Analysis: `analysis/02_dense_transition.ipynb`
