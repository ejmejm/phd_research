#!/bin/bash
./go.sh p_harm_rising UTILITY=curv PARENTS=harmonic EXPLORE=loss MATURITY=0 PROTECT=rising
./go.sh p_rising MATURITY=0 PROTECT=rising
echo batch11 done >> log.txt
