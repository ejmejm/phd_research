#!/bin/bash
# Mode switch 0 -> 1 at 200k, 400k steps, 600 LTUs, 1 seed: new method, old method, linear learner alone.
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
export PYTHONPATH=/home/edan/local_projects/phd_research
until grep -q "exec06 done" exec06.log; do sleep 10; done
F="grep -v -i warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU"
COMMON="OUT_GROW=1 WIDTH=600 N_SEEDS=1 N_STEPS=400000 SWITCH=200000 SWITCH_FILE=data_mode1_diff0.npz"
env $COMMON UTILITY=curv PARENTS=loss MATURITY=0 PROTECT=rising LABEL=sw_new /home/edan/miniconda3/envs/jax/bin/python run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU" >> log.txt
env $COMMON MATURITY=50000 LABEL=sw_old /home/edan/miniconda3/envs/jax/bin/python run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU" >> log.txt
env $COMMON LINEAR=1 LABEL=sw_lin /home/edan/miniconda3/envs/jax/bin/python run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU" >> log.txt
echo batch17 done >> log.txt
