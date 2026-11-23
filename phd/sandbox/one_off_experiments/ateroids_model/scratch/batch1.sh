#!/bin/bash
./go.sh p_random DUMP=1
./go.sh p_catalog DUMP=1 PARENTS=catalog
echo batch1 done >> log.txt
