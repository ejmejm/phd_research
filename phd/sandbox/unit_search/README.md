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
- `blocked_unit.ipynb` — a minimal test of whether a learned mediocre solution blocks a new unit: one target LTU on
  10 inputs, and a hidden unit reading all of them added before or after a linear readout of the inputs has
  converged, with network Autostep and SGD, and with up to 32 hidden units added after the readout.
