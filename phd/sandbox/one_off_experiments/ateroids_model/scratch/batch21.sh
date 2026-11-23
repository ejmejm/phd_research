#!/bin/bash
# error times the fraction a new LTU removes, from the LTUs tried there: pseudo-trial at the explained fraction
# (trials) or at a line fitted from explained fraction to outcome (trials_fit). 600 LTUs, 3 seeds, 400k.
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
O="OUT_GROW=1 WIDTH=600 N_SEEDS=3 N_STEPS=400000 UTILITY=curv MATURITY=0 PROTECT=rising"
bash go.sh x_trials $O PARENTS=trials &
bash go.sh x_trials_fit $O PARENTS=trials_fit &
wait
echo batch21 done >> log.txt
