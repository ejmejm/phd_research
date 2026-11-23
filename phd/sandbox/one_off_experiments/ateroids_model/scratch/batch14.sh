#!/bin/bash
until grep -q "^batch13 done" log.txt; do sleep 5; done
./go.sh l_progress UTILITY=curv PARENTS=progress EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising
./go.sh l_benefit UTILITY=curv PARENTS=benefit EXPLORE=loss N_STEPS=400000 MATURITY=0 PROTECT=rising
echo batch14 done >> log.txt
