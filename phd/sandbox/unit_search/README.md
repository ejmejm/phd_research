# Unit search

A network with a fixed connection budget learns which hidden units exist, not just which inputs each one reads: it
prunes whole units and generates new ones in the budget they free. Connection-level learning on its own is in
`phd/sandbox/feature_sifting/networks/`.

- `parallel_mnist_label_changes.ipynb` — unit search from an empty network on 4-task parallel MNIST with label
  changes, at the ideal network's connection budget.
