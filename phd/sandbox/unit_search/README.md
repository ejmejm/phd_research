# Unit search

A network with a fixed connection budget learns which hidden units exist, not just which inputs each one reads: it
prunes whole units and generates new ones in the budget they free. Connection-level learning on its own is in
`phd/sandbox/feature_sifting/networks/`.

- `parallel_mnist_label_changes.ipynb` — unit search from an empty network on 4-task parallel MNIST with label
  changes, at the ideal network's connection budget.
- `sparse_teacher.ipynb` — a deep, sparsely connected LTU target network (1,000 inputs, 3 × 20 hidden units, 100
  outputs): how fully connected networks scale with width, then unit search with units at depths 1–3 that read any
  lower depth.
- `sparse_teacher_dense_units.ipynb` — the same target, with unit search whose new units connect to everything
  before and after them, and connection pruning removes the connections that do not matter.
- `blocked_unit/` — steps from a minimal test toward the sparse target, in order:
  - `01_blocked_unit.ipynb` — whether a learned mediocre solution blocks a new unit: one target LTU on 10 inputs,
    and a hidden unit reading all of them added before or after a linear readout of the inputs has converged, with
    network Autostep and SGD, and with up to 32 hidden units added after the readout.
  - `02_blocked_unit_distractors.ipynb` — the same with 990 distractor inputs: a linear learner first, then hidden
    LTUs that read every input, 1 to 20 inputs sampled by the linear learner's weights, or only the target's 10.
  - `03_several_targets.ipynb` — the target is a sum of 1 to 8 target LTUs, each on its own 10 of the 1,000 inputs:
    32 LTUs added after the linear learner, reading inputs sampled by its weights, or one target LTU's inputs.
  - `04_feature_search.ipynb` — the same, with search: every 10k, 30k or 100k steps the 4 LTUs with the smallest
    output weights are replaced by LTUs reading newly sampled inputs.
  - `04a_fast_feature_learning.ipynb` — LTUs given the right inputs, with 8 target LTUs: how a higher initial
    step-size, initial weights from the linear learner, and adding the linear prediction back to their error speed up
    learning.
  - `04b_wrong_inputs.ipynb` — the same LTUs with 10 wrong inputs each: whether wrong inputs slow them, and whether
    seeded step-sizes or zero-crossing pruning help.
  - `04c_growing_features.ipynb` — LTUs that start from a few inputs and connect the input whose gradient stands out,
    starting from right or sampled inputs, with 8 target LTUs.
  - `04d_connected_to_everything.ipynb` — LTUs connected to every input, all weights at 0 with tiny step-sizes except a
    few chosen connections: whether Autostep raises the useful connections' step-sizes on its own.
  - `05_fast_feature_search.ipynb` — feature search with fast feature learning and growth: where the pieces of feature
    search come together.
  - `06_multiple_outputs.ipynb` — 05's growing LTUs with 10 outputs, each reading 4 of 8 target LTUs, and a shadow
    weight on one random output per LTU.
  - `07_two_hidden_layers.ipynb` — 06 with two hidden layers, in the learner (16 LTUs per depth, depth 2 growing over
    inputs and depth-1 LTUs, and depth-2 LTUs that barely read depth 1 moving to depth 1) and in the target network.
  - `08_shadow_weights_on_depth_2.ipynb` — 07 with depth-1 LTUs that can take their shadow weight on a depth-2 LTU,
    drawn at random or by usefulness, instead of on an output.
  - `summary.md` — what 01–05 found, in brief, and the directions not yet explored.
