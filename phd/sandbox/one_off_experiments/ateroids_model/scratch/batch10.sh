#!/bin/bash
for W in 300 1200 2400; do ./go.sh w_old_$W WIDTH=$W N_STEPS=400000; done
echo batch10 done >> log.txt
