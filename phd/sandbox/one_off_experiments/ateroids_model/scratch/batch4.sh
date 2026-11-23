#!/bin/bash
./go.sh p_harm_loss DUMP=1 PARENTS=harmonic EXPLORE=loss
./go.sh p_harm_uni DUMP=1 PARENTS=harmonic
echo batch4 done >> log.txt
