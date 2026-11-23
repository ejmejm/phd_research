#!/bin/bash
./go.sh p3_harm UTILITY=curv PARENTS=harmonic EXPLORE=loss N_SEEDS=3
./go.sh l_random N_STEPS=400000
./go.sh l_harm UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000
echo batch6 done >> log.txt
