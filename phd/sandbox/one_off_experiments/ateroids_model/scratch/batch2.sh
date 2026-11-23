#!/bin/bash
./go.sh p_benefit DUMP=1 PARENTS=benefit
./go.sh p_progress DUMP=1 PARENTS=progress
echo batch2 done >> log.txt
