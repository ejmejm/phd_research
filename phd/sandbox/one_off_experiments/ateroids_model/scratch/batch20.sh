#!/bin/bash
# Error times explained fraction (unit_search/output_prioritization notebook 4a) vs parents by loss, 600 LTUs, 3 seeds
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
O="OUT_GROW=1 WIDTH=600 N_SEEDS=3 N_STEPS=400000 UTILITY=curv MATURITY=0 PROTECT=rising"
bash go.sh x_loss $O PARENTS=loss &
bash go.sh x_explained $O PARENTS=explained &
wait
echo batch20 done >> log.txt
