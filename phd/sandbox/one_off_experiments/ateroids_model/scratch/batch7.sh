#!/bin/bash
./go.sh l_curv UTILITY=curv N_STEPS=400000
./go.sh l_harmabs PARENTS=harmonic EXPLORE=loss N_STEPS=400000
./go.sh l_harm_rising UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising
./go.sh l_harm_m100k UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=100000
./go.sh l3_random N_STEPS=400000 N_SEEDS=3
./go.sh l3_harm UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 N_SEEDS=3
echo batch7 done >> log.txt
