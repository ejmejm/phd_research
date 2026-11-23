#!/bin/bash
until grep -q "^batch12 done" log.txt; do sleep 10; done
./go.sh l_focus25 FOCUS=0.25 UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising
./go.sh l_focus75 FOCUS=0.75 UTILITY=curv PARENTS=harmonic EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising
echo batch13 done >> log.txt
