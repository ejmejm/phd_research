#!/bin/bash
./go.sh l3_harm_rising UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising N_SEEDS=3
./go.sh l3_harmabs_rising PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising N_SEEDS=3
echo batch12 done >> log.txt
