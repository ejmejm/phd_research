#!/bin/bash
for W in 300 1200 2400; do ./go.sh w_new_$W WIDTH=$W UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising; done
echo batch9 done >> log.txt
