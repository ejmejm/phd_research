#!/bin/bash
./go.sh l_harm_rising_oe UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising OF_ELIGIBLE=1
./go.sh l_harm_m100k_oe UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=100000 OF_ELIGIBLE=1
./go.sh l_harm_m50k_oe UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 OF_ELIGIBLE=1
echo batch8 done >> log.txt
