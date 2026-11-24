#!/bin/bash
./go.sh p_loss DUMP=1 PARENTS=benefit FOCUS=0 EXPLORE=loss
./go.sh p_progress_loss DUMP=1 PARENTS=progress EXPLORE=loss
./go.sh p_benefit_loss DUMP=1 PARENTS=benefit EXPLORE=loss
echo batch3 done >> log.txt
