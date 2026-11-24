# Unit Search Algorithm

The meta progression. Each folder of `../` is one progression on one part of the algorithm. Here, one notebook per
progression adds that progression's result to the algorithm. Each notebook compares the algorithm of the step before
(red) with the new algorithm (blue) on one problem: the problem of the notebook before, or that problem with one
change that shows why the new rule is needed.

1. `1_unit_search.ipynb` — the start: the algorithm of `../step_by_step/` on its deep sparse target network, against
   a fully connected network with network Autostep.
2. Pruning and depth (`../pruning_and_depth/`): to do.
3. Output prioritization (`../output_prioritization/`): to do.
4. Connection slots (`../connection_slots/`): to do.

**Code.** `algorithms.py` has one equinox class per step, and the static fields of each class are its
hyperparameters. The base class `Algorithm` holds network Autostep (`learn_step`) and the main loop (`run`), which
calls the same parts in every algorithm: `update_measures` in each learn step, `add_connections` then
`prune_connections` every `connection_period` steps (1k; PRUNE acts only every `prune_period` steps, 10k), and
`move_units`, `prune_units` and `add_units` every `unit_period` steps (10k). Each later class inherits from the class before it and overrides only the parts that change. The fully connected baseline uses the
same network and learn step, with fixed layer-to-layer connections. `problems.py` has the problems, and
`experiment.py` runs the algorithms over seeds and plots them.

`algorithms.py` computes the learn step with autograd, through the layers of `layers.py`: `ltu`, `linear` (shadow
weights in the backward pass) and `Standardize`, as in `../../sharables/network_autostep/`. `algorithms_manual.py` is
a backup of an earlier version with a hand-written backward pass, for comparison. That version matched the code of
`../step_by_step/6_sparse_target_network.ipynb`: on the same seeds, the errors of the first 12k steps were identical.

The current `UnitSearch` is the same algorithm in shorter code. SWAP exchanges the depths of two LTUs instead of their
slots, and each LTU stores its parent's index instead of shadow weights in its parent's weights. On the same network
state and key, each structure change gives exactly the result of the earlier version, except for the random draw of
new parents.

Notebook 1 takes about 7 minutes: 1M steps on the problem of `../step_by_step/`. The fully connected network takes 5
of them, because its LTU table holds a weight for every possible layer input.
