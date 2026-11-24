#!/bin/bash
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
export PYTHONPATH=/home/edan/local_projects/phd_research
PY=/home/edan/miniconda3/envs/jax/bin/python
LABEL=g2_w2400_mat50000_og OUT_GROW=1 MATURITY=50000 WIDTH=2400 N_SEEDS=3 $PY run.py 2>&1 | grep -v -i "warn\|autotun" >> log.txt
$PY capture.py 180000 200000 g2_w300_mat50000_og g2_w600_mat50000_og g2_w1200_mat50000_og g2_w2400_mat50000_og 2>&1 | grep -v -i warn >> log.txt
echo done >> log.txt
