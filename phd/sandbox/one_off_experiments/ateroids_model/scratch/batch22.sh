#!/bin/bash
# error times a line fitted from explained fraction to what new LTUs achieve for their parent (no per-output memory)
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
O="OUT_GROW=1 WIDTH=600 N_SEEDS=3 N_STEPS=400000 UTILITY=curv MATURITY=0 PROTECT=rising"
bash go.sh x_fit_line2 $O PARENTS=fit_line
echo batch22b done >> log.txt
