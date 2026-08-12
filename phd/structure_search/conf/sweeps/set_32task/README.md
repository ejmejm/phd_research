# set_32task — SET at 32 tasks, plus a batch-size study

Non-stationary parallel MNIST at **32 tasks**. Where `set_analysis_v2` swept
SET's design space at 8 tasks, this carries its single winner forward and spends
the compute on two new questions instead:

1. Does SET still beat a static sparse topology when the task count goes up 4x?
2. Does batching help, at a fixed budget of *samples*?

Everything is on the Comet project `set-32task-sweep`.

## What is fixed, and why

From the 8-task non-stationary results (`analysis/set_analysis_v2`, objective
`asymptotic_loss` summed over tasks):

| arm | best config | loss |
|---|---|---|
| SET, `epsilon` + `utility` | H=384, ζ=0.01, lr=2⁻⁶ | **9.26** |
| SET static (ζ=0) | H=384, lr=2⁻⁶ | 13.84 |
| SET, `epsilon` + `magnitude` | — | 15.41 |
| SET, `uniform_p` + `utility` | — | 13.26 |

So `init_mode=epsilon`, `prune_metric=utility` and `zeta=0.01` are **baked into
`base_set.yaml`**, not swept. `random_sparse` is dropped entirely: SET static
already beats it and is the tighter control (same ER init, same everything, only
the evolution removed), so it is the only baseline here.

**Budget and width.** `connection_budget = 4 * n_tasks * 794` (101,632 at n=32),
which holds the per-task budget fixed as tasks are added, and
`target_hidden_units = 48 * n_tasks` (1536 at n=32). Together these reproduce the
8-task winner exactly at any task count: ε ≈ 3.57 and ~62 W1 weights per hidden
unit. `target_hidden_units` is a *density* knob, not a capacity knob — the budget
sets capacity, and H decides how thinly it is spread.

**The one swept axis in `01_main` is the learning rate.** The 8-task winner sat
at the top edge of its LR range (2⁻⁶ of {2⁻¹⁰..2⁻⁶}), so the grid here is shifted
up an octave to {2⁻⁷, 2⁻⁶, 2⁻⁵}. If the minimum lands on 2⁻⁵ again, extend before
trusting `02_batch_size`.

## The sample timeline

`base_set.yaml` defines the run on a **sample** axis and derives every
step-based knob by dividing through by `batch_size`:

| knob | derivation | B=1 | B=4 | B=16 | B=64 |
|---|---|---|---|---|---|
| `train.total_steps` | 512,000 / B | 512000 | 128000 | 32000 | 8000 |
| `train.log_freq` | 1024 / B | 1024 | 256 | 64 | 16 |
| `set.evolve_frequency` | 512 / B | 512 | 128 | 32 | 8 |
| `task.permute_period` | 2048 / (n_tasks · B) | 64 | 16 | 4 | 1 |
| `set.utility_decay` | 0.998^B | 0.99800 | 0.99202 | 0.96848 | 0.87974 |

Consequences worth knowing:

- All four batch sizes see the **same 512,000-sample stream with the same drift
  schedule** (each task re-permuted every 2048 samples, matching the 2000 used at
  8 tasks) and the same amount of SET churn per sample. The *only* difference is
  the number of optimizer updates. That makes the study a clean read of "same
  data, fewer but better-estimated updates" — plot against **samples, not steps**.
- Every run emits exactly **500 log points, 1024 samples apart**, whatever the
  batch size, so learning curves overlay directly on a sample axis.
- `log_freq` stays divisible by `evolve_frequency` (asserted in
  `run_experiment`) for powers of two only. **B=64 is the ceiling**: at B=128
  `permute_period` would floor to 0 and the drift would switch off silently.

Batching itself needs no new code — `train.batch_size` is already a vmapped
minibatch whose mean loss makes it exactly equivalent to averaging B per-sample
gradients. `tests/test_set_batching.py` asserts that identity (a batched step ==
the mean of the B single-sample steps, for the weights *and* the utility traces),
plus that `permute_period` counts steps rather than samples, which is what the
1/B scaling above relies on. The identity is exact in real arithmetic; on GPU the
default TF32 matmul precision makes the two sides differ by ~1e-4 relative, so
the test pins `jax_default_matmul_precision='highest'` to measure the algebra
rather than the accelerator. Real runs keep the default.

## Sweeps

`command` points at `conf/sweeps/set_32task/base_set` and is resolved at runtime
relative to `experiments/`. Each trial vmaps 5 seeds (0–4).

| Config | Sweep ID | Trials | Grid |
|--------|----------|--------|------|
| `01_main/sweep/set` | `2ce99280dab74dd29c5b0c71d92e7990` | 3 | 3 LR |
| `01_main/sweep/set_static` | `43dd6aa99fe74da698e8accb771c0806` | 3 | 3 LR, ζ=0 |
| `02_batch_size/sweep/set` | not yet created | 12 | 4 B × 3 LR |
| `02_batch_size/sweep/set_static` | not yet created | 12 | 4 B × 3 LR, ζ=0 |

Resource plan and sbatch commands are under [Sweep Runs](#sweep-runs) below.

`02_batch_size` is deliberately **not** created yet: its LR grid
({2⁻⁶, 2⁻⁵, 2⁻⁴}) assumes `01_main` lands at 2⁻⁶ or 2⁻⁵. Read `01_main` first,
re-centre the grid if the winner is 2⁻⁷, then:

```bash
comet_sweep --config \
  conf/sweeps/set_32task/02_batch_size/sweep/set.yaml \
  conf/sweeps/set_32task/02_batch_size/sweep/set_static.yaml
```

### Cost

32 tasks is **far** more expensive per step than 8 was: `input_dim` goes 6272 →
25088 and H goes 384 → 1536, so the dense `(output_dim, buffer_size)` =
(320, 26624) output-weight matmul dominates, and it is memory-bound at B=1.
Measured on the RTX 4080, n=32, 5-seed vmap:

| B | steps/s | samples/s | steps for 512k samples | per trial (4080) | est. MIG |
|---|---|---|---|---|---|
| 1 | 277 | 277 | 512,000 | ~31 min | ~78 min |
| 4 | 125 | 500 | 128,000 | ~17 min | ~43 min |
| 16 | 44.9 | 718 | 32,000 | ~12 min | ~30 min |
| 64 | 12.4 | 794 | 8,000 | ~11 min | ~27 min |

Batching buys throughput as well as gradient quality — 2.9x the samples/s at
B=64 — because that dense matmul is amortized over the batch. So the large-batch
cells are the *cheap* ones; the B=1 cells dominate the wall-clock.

For reference the 8-task sweeps ran ~6000 steps/s on this GPU and ~2200 on an
H100 MIG 1g.10gb slice; the MIG column above assumes the same ~2.5x factor.

No OOM at any batch size; peak host RSS 3.6 GB, hence `--mem=16G` rather than the
usual 6G.

## Sweep Runs

Hrs/trial is the MIG estimate; total hrs includes the standard 1.4x buffer. All
`01_main` trials are B=1, so 1.3 h/trial; `02_batch_size` averages over its four
batch sizes (1.3 + 0.72 + 0.5 + 0.45 = 2.97 h per LR triple / 4 ≈ 0.74 h/trial).
Trials exceed 1 h, so these use the 12 h slot — a job must never be killed
mid-trial.

| Config | Sweep ID | Trials | Steps/sec | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|-----------|----------|----------|
| `01_main/sweep/set` | `2ce99280dab74dd29c5b0c71d92e7990` | 3 | ~110 (B=1) | 1.3 | 5.5 | 12h | 1 |
| `01_main/sweep/set_static` | `43dd6aa99fe74da698e8accb771c0806` | 3 | ~110 (B=1) | 1.3 | 5.5 | 12h | 1 |
| `02_batch_size/sweep/set` | not yet created | 12 | 110–320 | 0.74 | 12.4 | 12h | 2 |
| `02_batch_size/sweep/set_static` | not yet created | 12 | 110–320 | 0.74 | 12.4 | 12h | 2 |

6 jobs total, far inside the rate limits.

```bash
# --- 01_main. Launch the SET arm first, confirm from its log that a trial is
#     training (n_tasks=32 has never run on CC), then launch the static arm. ---
sbatch --array=1-1 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=16G --time=12:00:00 \
  launch_comet_agent.sbatch -s 2ce99280dab74dd29c5b0c71d92e7990 \
  -p $HOME/scratch/phd_research/phd/structure_search

sbatch --array=1-1 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=16G --time=12:00:00 \
  launch_comet_agent.sbatch -s 43dd6aa99fe74da698e8accb771c0806 \
  -p $HOME/scratch/phd_research/phd/structure_search

# --- 02_batch_size, once its sweeps exist (fill in the IDs) ---
for ID in <02_batch_size/sweep/set ID> <02_batch_size/sweep/set_static ID>; do
  sbatch --array=1-2 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=16G --time=12:00:00 \
    launch_comet_agent.sbatch -s $ID -p $HOME/scratch/phd_research/phd/structure_search
done
```

Each worker pulls trials until the sweep is exhausted or the job ends, so
`01_main` finishes in ~3.9 h of wall-clock on one worker per arm.

`--mem=16G` rather than the 8G used at 8 tasks: the host holds one log period of
data per seed, which is 1024 × 32 × 784 floats × 5 streams ≈ 500 MB, and peak
RSS measured 3.6 GB.

## Smoke test

`n_tasks=2` scales every derived quantity down (H=96, budget=6352, ε≈3.57 — the
same per-unit density) and runs in a couple of minutes:

```bash
python experiments/train_set.py --config-path=../conf/sweeps/set_32task --config-name=base_set \
  task.n_tasks=2 samples.total=20480 train.batch_size=1 \
  mlflow=false log_individual_seeds=false 'optimizer.learning_rate=${eval:2**-6}'
```

Swap `train.batch_size` for 4/16/64 to check the derived timeline; the printed
`SET sizing` line should report `hidden_units` exactly equal to
`48 * n_tasks` and `total ~= connection_budget`.

## Analysis

`analysis/set_32task/set_32task_analysis.ipynb` — LR sensitivity and learning
curves for `01_main`, then per-batch-size LR sensitivity and sample-axis learning
curves for `02_batch_size`.
