#!/bin/bash
./go.sh p_curv DUMP=1 UTILITY=curv
./go.sh p_curv_harm DUMP=1 UTILITY=curv PARENTS=harmonic EXPLORE=loss
echo batch5 done >> log.txt
