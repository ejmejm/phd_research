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

# A short run: 4-task multi-MNIST, block-sparse, 2000 steps on CPU.
python experiments/train.py device=cpu train.total_steps=2000 train.log_freq=500
```

MNIST downloads automatically on first run. To log to Comet, set
`COMET_API_KEY` and `COMET_WORKSPACE` in your environment and pass
`comet_ml=true`.

## Repository layout

| Path | What lives here |
|---|---|
| `src/multi_mnist/` | The library: problem, models, algorithms, training loop |
| `experiments/` | `train.py`, its Hydra configs, and the sweep definitions |
| `analysis/` | One notebook per figure, plus the downloaded run data |
| `figures/` | `paper/` holds the published figures, `generated/` what you re-run |

## Reproducing the figures

Each figure comes from one sweep family and one notebook:

| Figure | Sweep | Notebook |
|---|---|---|
| 1 — matched connectivity, stationary and non-stationary | `experiments/sweeps/01_matched_and_scaling/` | `analysis/01_matched_and_scaling.ipynb` |
| 2 — scaling the dense network, 1× to 64× | `experiments/sweeps/01_matched_and_scaling/` | `analysis/01_matched_and_scaling.ipynb` |
| 3 — the dense transition | `experiments/sweeps/02_dense_transition/` | `analysis/02_dense_transition.ipynb` |
| 4 — scaling the problem and the network | `experiments/sweeps/03_task_scaling/` | `analysis/03_task_scaling.ipynb` |

There are two ways to get the data the notebooks read.

**Run the sweeps yourself.** `experiments/sweeps/README.md` has the exact
commands, in order, with the compute each one needs. This is the full
reproduction path and takes GPU-days.

**Download ours.** Unpack the archive from
`<ARCHIVE_URL>` into `analysis/data/`, then run the notebooks directly.

Every notebook opens with a `download_project(...)` call, commented out, that
re-fetches the results from Comet if you have access to the project.

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
| `static` | Connectivity never changes | Figures 1, 2, 4 |
| `dense_transition` | Trains block-sparse, then makes the network dense at a chosen step | Figure 3 |
| `set` | Sparse Evolutionary Training: prune the smallest weights, regrow at random | Follow-up work |
| `deep_r` | DEEP-R: L1 + Langevin noise, prune on sign flip, regrow at random | Follow-up work |

`set` and `deep_r` each have two implementations, chosen by `model.type`:
`algorithms/{set,deep_r}.py` for `padded_mlp` and `algorithms/dynamic/` for
`dynamic_network`. They are the same algorithm on different storage; keep them
in step, or record in the config which one produced a result.

SET and DEEP-R are not in the paper. They are here because the paper argues
that learning connectivity is the promising direction, and this is the
harness to test that in.

### Adding an algorithm

Write one file in `src/multi_mnist/algorithms/` implementing
`ConnectivityAlgorithm`, and add a line to `build_algorithm`. The interface
has two hooks, and which one you need depends on the method:

- `event` runs **inside** the jitted, vmapped scan every `event_period` steps.
  It is traced, so it must be shape-stable. This is where prune-and-regrow
  belongs. `algorithms/set.py` is the worked example.
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

Notebooks are committed with outputs cleared, which matters more than it
sounds: a single traceback embeds absolute paths and a username. Before
committing:

```bash
jupyter nbconvert --clear-output --inplace analysis/*.ipynb
```

## Citation

```bibtex
@inproceedings{meyer2026connectivity,
  title     = {Connectivity, Credit Assignment, and the Speed of Learning},
  author    = {Meyer, Edan and Freeman, Andrew and Sutton, Richard S.},
  booktitle = {Continual Reinforcement Learning Workshop at RLC},
  year      = {2026},
}
```
