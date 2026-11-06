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
- `demo/` — a demo of unit search on a small random sparse, locally connected target network (33 inputs, 2 × 7 LTUs,
  7 outputs):
  - `sparse_target_demo.ipynb` — unit search against fully connected networks (SGD, Adam, batched Adam, network
    Autostep), each at its best step-size.
  - `record.py` — records one run of unit search (the notebook's code) to `data.js`, for `index.html`, a playback
    of the network as it learns (2D, layers as rows), beside the target network, with its RMSE.
- `step_by_step/` — unit search step by step, from a new unit blocked in a trained network to the deep sparse target
  network: six short notebooks, each with one question, the simplest experiment that answers it, and a simple fix. Its
  README has the method in pseudocode and a log of options to improve each part.
