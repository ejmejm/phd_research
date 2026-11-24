#!/bin/bash
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
export PYTHONPATH=/home/edan/local_projects/phd_research
O="OUT_GROW=1 WIDTH=600 N_SEEDS=3 N_STEPS=400000 UTILITY=curv PARENTS=loss MATURITY=0 PROTECT=rising"
env $O DATA_FILE=data_mode0_diff3.npz LABEL=fr3_d3_new /home/edan/miniconda3/envs/jax/bin/python run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU" >> log.txt
env $O SWITCH=200000 ALIGN=1 SWITCH_FILE=data_mode0_diff3.npz LABEL=sw3_d3_new /home/edan/miniconda3/envs/jax/bin/python run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU" >> log.txt
echo batch19 done >> log.txt
