#!/bin/bash
./go.sh l3_lossonly UTILITY=curv PARENTS=benefit FOCUS=0 EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising N_SEEDS=3
for W in 300 1200 2400; do ./go.sh w_loss_$W WIDTH=$W UTILITY=curv PARENTS=benefit FOCUS=0 EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising; done
echo batch16 done >> log.txt
