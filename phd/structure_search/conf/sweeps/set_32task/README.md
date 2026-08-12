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
up an octave to {2⁻⁷, 2⁻⁶, 2⁻⁵}. **Result: 2⁻⁶ for both arms** — interior, so no
extension was needed and `02_batch_size` could be anchored on it.

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
| `02_batch_size/sweep/set` | `0e0e4c42982043699da8ac33778491bd` | 16 | 4 B × 4 LR |
| `02_batch_size/sweep/set_static` | `480e38591b4040a684f133765e815b0e` | 16 | 4 B × 4 LR, ζ=0 |

Resource plan and sbatch commands are under [Sweep Runs](#sweep-runs) below.

`01_main` picked **lr = 2⁻⁶ for both arms**, interior to its
{2⁻⁷, 2⁻⁶, 2⁻⁵} grid — so no extension was needed. `02_batch_size` is anchored
there and runs three octaves up, {2⁻⁶ … 2⁻³}: both the lower gradient variance of
a large batch and its 1/B update budget push the optimum up, never down, so 2⁻⁶
is the floor; and sqrt-scaling puts B=64's optimum at 8× the B=1 step, i.e. 2⁻³,
the top of the grid. The optimum should therefore be interior at both ends.

Because LR is re-swept at every batch size (B=1 included), `01_main`'s winner
only sets where the grid is centred — the batch study re-measures its own B=1
reference rather than inheriting one.

### Cost

32 tasks is **far** more expensive per step than 8 was: `input_dim` goes 6272 →
25088 and H goes 384 → 1536, so the dense `(output_dim, buffer_size)` =
(320, 26624) output-weight matmul dominates. At B=1 it is *launch*-bound rather
than bandwidth-bound (see [Sweep Runs](#sweep-runs)). Measured on the RTX 4080,
n=32, 5-seed vmap:

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

**Measured on the MIG: 155 steps/s at B=1** (job 19699556, 2026-08-12), i.e.
0.92 h/trial — better than the 110 steps/s extrapolated from the 4080. The MIG is
only 1.8x slower than the 4080 rather than the 2.5x seen at 8 tasks, which fits
the launch-bound picture below: when you are latency-limited, a smaller slice
costs less than its SM count suggests. Other batch sizes below are the 4080
measurements scaled by the same 0.56 factor.

**Sized for wall-clock, not allocation efficiency**: one worker per trial, so all
trials run concurrently and a sweep finishes in the time of a single trial rather
than the sum. The 3 h slot also carries higher scheduling priority than 6 h/12 h,
and one 0.92 h trial fits it with 3x headroom.

| Config | Sweep ID | Trials | Steps/sec | Hrs/trial | Total hrs | Job time | Num jobs |
|--------|----------|--------|-----------|-----------|-----------|----------|----------|
| `01_main/sweep/set` | `2ce99280dab74dd29c5b0c71d92e7990` | 3 | 155 (measured) | 0.92 | 2.8 | 3h | 3 |
| `01_main/sweep/set_static` | `43dd6aa99fe74da698e8accb771c0806` | 3 | 155 (measured) | 0.92 | 2.8 | 3h | 3 |
| `02_batch_size/sweep/set` | `0e0e4c42982043699da8ac33778491bd` | 16 | 7–155 | 0.32–0.92 | 11.8 | 3h | 16 |
| `02_batch_size/sweep/set_static` | `480e38591b4040a684f133765e815b0e` | 16 | 7–155 | 0.32–0.92 | 11.8 | 3h | 16 |

6 jobs for `01_main`, 32 for `02_batch_size` — inside the 50-job cap.

Per-trial MIG times for `02_batch_size`: B=1 0.92 h, B=4 0.51 h, B=16 0.35 h,
B=64 0.32 h. One worker per trial puts its wall-clock at ~0.92 h too, set by the
slowest (B=1) cells.

A larger MIG slice or a full H100 would **not** speed up the B=1 trials. Each
step reads only ~170 MB across the 5 vmapped seeds; at the measured 277 steps/s
on the 4080 that is ~47 GB/s against ~717 GB/s available, so the scan is
launch/latency-bound on many tiny sequential steps, not compute- or
bandwidth-bound. Extra SMs buy nothing — concurrency across trials is the lever.

```bash
# --- 01_main: 6 jobs, one trial each, all concurrent -> ~1.3 h + queue ---
for ID in 2ce99280dab74dd29c5b0c71d92e7990 43dd6aa99fe74da698e8accb771c0806; do
  sbatch --array=1-3 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=16G --time=03:00:00 \
    launch_comet_agent.sbatch -s $ID -p $HOME/scratch/phd_research/phd/structure_search
done

# --- 02_batch_size: 32 jobs, one trial each -> ~0.92 h (set by the B=1 cells) ---
for ID in 0e0e4c42982043699da8ac33778491bd 480e38591b4040a684f133765e815b0e; do
  sbatch --array=1-16 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=16G --time=03:00:00 \
    launch_comet_agent.sbatch -s $ID -p $HOME/scratch/phd_research/phd/structure_search
done
```

`--mem=16G` overrides the launcher's hardcoded `--mem=4G` (command-line sbatch
options win over in-script directives). Logs land in `~/scratch/job_outputs/`.

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
