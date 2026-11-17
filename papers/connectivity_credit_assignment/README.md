# Connectivity, Credit Assignment, and the Speed of Learning

Code for the paper, and the platform its follow-up work builds on.

The connectivity of a neural network shapes how backpropagation assigns
credit, which in turn determines how fast it learns. We introduce
**multi-MNIST**, a multi-task, non-stationary testbed that isolates
connectivity's effect on credit assignment, and show that poor connectivity
slows learning by misassigning credit — which translates to worse performance
in non-stationary problems. Scaling the network cannot solve this, and past a
point makes it worse.

---

## Quick start

```bash
pip install -e ".[analysis]"

# A short GPU run: 4-task multi-MNIST, block-sparse, 2000 steps.
python experiments/train.py device=cuda train.total_steps=2000 train.log_freq=500
```

MNIST downloads automatically on first run. To log to Comet, set
`COMET_API_KEY` and `COMET_WORKSPACE` in your environment and pass
`comet_ml=true`.

## Repository layout

- `src/multi_mnist/` contains the models, algorithms, and training loop.
- `experiments/sweeps/` contains the experiment configurations and run records.
- `analysis/` contains the analyses and figure exports.
- `figures/generated/` contains regenerated figures and remains ignored by Git.
- `figures/paper/` retains the original paper figures, including the dense-width comparison.
- `latex_paper/` is a separate Overleaf Git checkout. Commit and push its changes independently.

## Reproducing the figures

The current manuscript uses these analyses:

- Figure 1 and Appendix C use `analysis/06_matched_width_and_permutations.ipynb`.
- Figure 2 uses `analysis/02_dense_transition.ipynb`.
- Figure 3 uses `analysis/03_task_scaling.ipynb`.
- Figure 4 uses `analysis/04_connectivity_methods_sensitivity.ipynb`.
- Appendix D uses `analysis/07_optimizers_batch_size.ipynb`.

Retained experiments outside the current manuscript:

- `analysis/01_matched_and_scaling.ipynb` contains the original matched-budget and dense-width comparisons.
- `analysis/05_dense_width_controls.ipynb` contains stationary learning curves across all dense widths, non-stationary curves, and the wider-dense permutation control.
- Notebook 06 also retains the dense-transition step-size-switch control.

Run notebooks from `analysis/`. Download Comet exports with the commented
`download_project(...)` calls when needed. Exports and per-seed caches remain
in the ignored `analysis/data/` directory. The original MLflow analyses in
notebooks 05 and 06 also need the exports at
`phd/structure_search/analysis/weight_pruning/data/` in the parent repository.

`experiments/sweeps/README.md` records the original experiment protocol.
Each family README records its configurations and later additions.
Historical figure numbers in those records refer to the original manuscript.
No new training run is needed for the saved figure proposal below.

## The problem

A multi-MNIST sample is `n` MNIST images concatenated into one input of
dimension `784n`, with `n` labels. The loss splits the network's `10n` outputs
into `n` groups and sums a softmax cross-entropy per group.

Because the `n` sub-problems are drawn independently, **the ideal connectivity
is known in advance**: `n` independent sub-networks, no cross-task
connections. That is what makes the problem useful — it tests the effect of
good connectivity without needing a method that finds it.

In the non-stationary variant, every `permute_period` steps one task is chosen
at random and its label mapping is replaced by a random permutation. Each
event hits one task, so the expected interval between re-permutations of a
*given* task is `permute_period × n_tasks`; the sweeps set
`permute_period = C / n_tasks` to hold that fixed as `n` grows.

## Models and algorithms

Connectivity is decided by a **model** (how connections are stored) and an
**algorithm** (how they change during training).

**Models** (`model.type`):

- `padded_mlp` — a masked dense matrix, and what every experiment here runs
  on. Sparsity is a mask, not a sparse data structure, so a sparse network
  costs exactly what a dense one costs. That is the right choice for
  matched-parameter comparisons and the wrong one for scale. Its
  `init_strategy` picks the pattern: `dense`, `block_sparse`, or `sparse` (an
  Erdős–Rényi topology, which is what SET and DEEP-R evolve from).
- `dynamic_network` — preallocated per-unit sparse storage with a custom VJP,
  so connectivity can change without recompiling. Setting `model.type` to it
  selects the SET and DEEP-R implementations in `algorithms/dynamic/`. Use it
  when the network is large enough that the dense matmul is wasteful; the
  trade is that the connection budget is then preserved per unit rather than
  per layer, and `path_purity` is unavailable.
- `BlockSparseMLP` — `n` sub-MLPs as batched tensors. Not wired into
  `train.py`; import it when you want block-sparse results quickly and don't
  need a matched-cost dense comparison.

**Algorithms** (`algorithm.name`):

| Name | What it does | Used by |
|---|---|---|
| `static` | Connectivity never changes | Fixed-connectivity baselines |
| `dense_transition` | Trains block-sparse, then makes the network dense at a chosen step | Figure 2 |
| `set` | Sparse Evolutionary Training, or Continual SET with contribution utilities | Figure 4 |
| `deep_r` | DEEP-R: L1 + Langevin noise, prune on sign flip, regrow at random | Figure 4 |

`set` and `deep_r` each have two implementations, chosen by `model.type`:
`algorithms/{set,deep_r}.py` for `padded_mlp` and `algorithms/dynamic/` for
`dynamic_network`. They are the same algorithm on different storage; keep them
in step, or record in the config which one produced a result.

Continual SET uses `prune_metric: utility`, `utility_reset: median`,
and `regrow: layer_lecun` on `padded_mlp`. Its final configuration is
`experiments/sweeps/04_connectivity_methods/best_2m/continual_set_median_global.yaml`.
The preceding utility sweep used `row_lecun`; that implementation remains
available to reproduce the sweep.

### Adding an algorithm

Write one file in `src/multi_mnist/algorithms/` implementing
`ConnectivityAlgorithm`, and add a line to `build_algorithm`. The interface
has three hooks, and which one you need depends on the method:

- `event` runs **inside** the jitted, vmapped scan every `event_period` steps.
  It is traced, so it must be shape-stable. This is where prune-and-regrow
  belongs. `algorithms/set.py` is the worked example.
- `step_update` also runs **inside** the scan, but on every step, and may
  modify the model. It is for methods whose weight dynamics run faster than
  their structural bookkeeping. `algorithms/deep_r.py` is the example.
- `on_log_period` runs **on the host** between log periods, with concrete
  values. Use it for one-shot interventions where a Python flag is clearer
  than a traced counter. `algorithms/dense_transition.py` is the example.

`base.py` documents the rest, including why `step_hook` receives both the
pre- and post-update model.

### When to write a second entry point instead

`experiments/train.py` covers a run that is: sample a batch, take a gradient
step, occasionally change connectivity, log. That was every training script in
the original codebase — five of them, ~1000 lines each, with the same loop
skeleton and differences only in the model and the structure update. Those
differences are now `model.type` and `algorithm.name`.

Write a separate entry point when the *loop* changes shape, not when the
method does. Signs it has:

- the step needs more than one forward pass, or a second model (a target
  network, a teacher, a meta-gradient through the update);
- structure changes need host interaction mid-scan, so they cannot be traced;
- the problem is not multi-MNIST, so the data stream and loss change too.

Forking is cheap and not a failure: `training.py` exposes `train_step`,
`build_scan_log_period`, and `run_experiment` separately, so a new entry point
can reuse the parts that still apply and replace the loop. Diagnostics and
small-scale prototypes belong in a notebook rather than either.

## Personal information

Four places, and no tooling:

- `pyproject.toml` — author name and email.
- `README.md` — the citation block below.
- **Comet workspace** — never in a file. `comet_ml` reads `COMET_WORKSPACE`
  from your environment or `~/.comet.config`.
- **Comet project names** — literal strings at the top of each sweep config
  and each notebook (`paper-weight-pruning-*`). These are the projects the
  published runs live in, so the notebooks read the paper's data as shipped.
  Change them and your runs go somewhere else.

Commit notebooks with outputs and execution counts cleared. Edit only the
relevant JSON fields and cells. Preserve notebook metadata and unrelated source.
Read each notebook from disk immediately before every edit.

## Citation

```bibtex
@inproceedings{meyer2026connectivity,
  title     = {Connectivity, Credit Assignment, and the Speed of Learning},
  author    = {Meyer, Edan and Freeman, Andrew and Sutton, Richard S.},
  booktitle = {Continual Reinforcement Learning Workshop at RLC},
  year      = {2026},
}
```

## Idea for the next revision

Keep the revised Figure 1, which compares networks with equal hidden-unit counts.
Then consider a figure with two panels, based on the original Figure 2.
Keep the block-sparse network fixed and increase the dense network width.

- Stationary targets. More width lets dense connectivity match and then exceed the fixed block-sparse reference.
- Non-stationary targets. More width fails to close the gap and eventually reduces accuracy within the tested range.

The question is whether extra width can compensate for slower adaptation.
Equal width establishes that the dense network can represent the block-sparse solution.
It does not establish that the dense network can learn that solution equally fast.
The stationary panel makes the different effect of width under non-stationarity clear.

Possible paper text:

> The dense network can represent the block-sparse solution, but adapts more slowly. Could additional width compensate for this difference? We increase the dense network's width while keeping the block-sparse network fixed, and compare their performance on stationary and non-stationary problems.

Place this result before the dense-transition experiment, then show how the cost changes with the number of subproblems.
Preserve the dense-width experiments and their stationary learning curves for this possible revision.
This is a saved proposal, not a change to the current manuscript.
