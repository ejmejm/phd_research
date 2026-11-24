#!/bin/bash
./go.sh l_lossonly UTILITY=curv PARENTS=benefit FOCUS=0 EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising
./go.sh l_random_rising UTILITY=curv N_STEPS=400000 MATURITY=0 PROTECT=rising
echo batch15 done >> log.txt
