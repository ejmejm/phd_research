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

## `sweep/forward_only.yaml` — which direction does the damage

The main sweep shows that the flip hurts, not why. A new cross-task connection
hurts in two directions at once:

| | what changes |
|---|---|
| forward | another task's signal reaches this unit's activation, and this output |
| backward | another task's error reaches this unit's incoming weights |

`algorithm.forward_only_distractors: true` removes the second one. The
connections the flip adds still run in the forward pass and still learn their
own weights from `dL/do * h`; what they no longer do is return error to the
hidden layer, so every unit's `W1` row is still trained on its own task's loss
alone. Whatever degradation survives is the forward channel; the gap to the
ordinary transition at the same step-size is the backward one.

Only `W2` needs the intervention. A new `W1` connection's backward path ends
at an input, which is not a parameter, so `W1` distractors are already
forward-only. It is implemented as a `stop_gradient` on the hidden activations
in the forward-only half of the `W2` matmul, pinned at setup by
`models.padded_mlp.freeze_gradient_mask` — before the flip, the mask covers
everything and the run is bit-identical to the ordinary one.

Not a sweep: two cells, the two step-sizes Figure 3 already plots the
transition at. 30 seeds each, as six 5-seed blocks — 12 trials. The main sweep
runs 15 seeds per trial, which does not fit on a 16 GB card once a Comet trial
is alongside it; the per-log-period batch alone is 1000 × 15 × 16 × 784 × 4 B
= 752 MB. `stack_per_seed` rejoins any number of blocks.

| step-size | why |
|---|---|
| 2⁻¹⁰ | the transition's own best, from the main sweep |
| 2⁻⁸ | block-sparse's best — the curve that isolates the flip from re-tuning |

```bash
comet_sweep -c sweeps/02_dense_transition/sweep/forward_only.yaml
```

Analysis: `analysis/02_dense_transition.ipynb`
