#!/bin/bash
# go.sh LABEL [VAR=VALUE ...]: one run of run.py with those settings (OUT_GROW=1 MATURITY=50000 WIDTH=600 N_SEEDS=1 by
# default), then its gap closed; both appended to log.txt.
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
export PYTHONPATH=/home/edan/local_projects/phd_research
PY=/home/edan/miniconda3/envs/jax/bin/python
L=$1; shift
env OUT_GROW=1 MATURITY=50000 WIDTH=600 N_SEEDS=1 "$@" LABEL=$L $PY run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|allocation summary\|TF_GPU" >> log.txt
N=$(env N_STEPS=200000 "$@" bash -c 'echo $N_STEPS')
$PY capture.py $((N * 9 / 10)) $N $L 2>&1 | grep -v -i "warn\|pinned\|allocator\|bfc\|allocation summary\|TF_GPU" >> log.txt
