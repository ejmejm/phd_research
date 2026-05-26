# block_sparse_dense_fill — Comet sweeps for Compute Canada

Tests whether filling a converged block_sparse MLP into a dense MLP at a
chosen training step preserves performance, vs. (a) dense-from-scratch
with cross-task weights starting at 0, and (b) the block_sparse baseline.

Per-trial output: 30 atomic seeds vmapped (split into two 15-seed halves
on `main` to fit a 10 GB MIG slice). End-of-trial uploads
`per_seed_losses.npy` and `per_seed_accs.npy` to Comet for true
bootstrap-CI plotting in the analysis notebook.

## Sweep Runs

| Config | Sweep ID | Trials | Steps/sec (4080) | Hrs/trial (H100 est.) | Total H100 hrs | Job time | Num jobs |
|--------|----------|--------|------------------|-----------------------|----------------|----------|----------|
| main | `dc6d3df68d25479999b72a32af29c275` | 30 | 370 (15-seed vmap) | ~0.23 | ~7 | 3h | 4 |

`small.yaml` is intentionally **not** on CC — local-only.

```bash
# main test job (1h, 1 worker)
sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=10G --time=01:00:00 \
  launch_comet_agent.sbatch \
  -s dc6d3df68d25479999b72a32af29c275 \
  -p $HOME/scratch/phd_research/phd/structure_search

# main final array (3h, 4 parallel workers)
sbatch --array=1-4 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=10G --time=03:00:00 \
  launch_comet_agent.sbatch \
  -s dc6d3df68d25479999b72a32af29c275 \
  -p $HOME/scratch/phd_research/phd/structure_search
```

## Grid

- `model.dense_fill_step`: `[0, 250_000, 600_000]` — `600_000 > total_steps=500_000` is the "never fill" sentinel (block_sparse baseline).
- `optimizer.learning_rate`: `2**-{11, 10, 9, 8, 7}`.
- `seed_offset`: `[0, 15]` → atomic 15-seed list built via `${eval:${seed_offset} + N}`.
