# n_task_scaling — Comet sweeps for Compute Canada

Joint scan over `task.n_tasks ∈ {2,4,8,16,32}` × `model.initial_hidden_units ∈ {4..512}` × 4 LRs for both `block_sparse` and `dense` init strategies. Repeats the existing mlflow sweeps with 5 atomic seeds vmapped per trial (instead of 1) and uploads per-seed curve arrays as a comet asset for CI plotting.

`block_sparse` cells where `hidden < n_tasks` (12 cells × 4 LR = 48 trials) fail-fast by design.

## Sweep Runs

| Config | Sweep ID | Trials | Steps/sec (4080, est. 5-seed vmap) | Hrs/trial (H100 est.) | Total H100 hrs | Job time | Num jobs |
|--------|----------|--------|-------------------------------------|-----------------------|----------------|----------|----------|
| block_sparse | `bf92bf7c1ced4535823e892b8049089d` | 160 | varies (~5k–20k) | ~0.08 avg | ~13 | 3h | 10 |
| dense | `b21316aca20d4aa7b60d05f03d59babc` | 160 | varies (~5k–20k) | ~0.08 avg | ~13 | 3h | 10 |

```bash
# block_sparse test job (1h, 1 worker)
sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=10G --time=01:00:00 \
  launch_comet_agent.sbatch \
  -s bf92bf7c1ced4535823e892b8049089d \
  -p $HOME/scratch/phd_research/phd/structure_search

# dense test job (1h, 1 worker)
sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=10G --time=01:00:00 \
  launch_comet_agent.sbatch \
  -s b21316aca20d4aa7b60d05f03d59babc \
  -p $HOME/scratch/phd_research/phd/structure_search

# block_sparse final array (3h, 10 parallel workers)
sbatch --array=1-10 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=10G --time=03:00:00 \
  launch_comet_agent.sbatch \
  -s bf92bf7c1ced4535823e892b8049089d \
  -p $HOME/scratch/phd_research/phd/structure_search

# dense final array (3h, 10 parallel workers)
sbatch --array=1-10 --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --cpus-per-task=1 --mem=10G --time=03:00:00 \
  launch_comet_agent.sbatch \
  -s b21316aca20d4aa7b60d05f03d59babc \
  -p $HOME/scratch/phd_research/phd/structure_search
```
