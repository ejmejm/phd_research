#!/bin/bash
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model/scratch
export PYTHONPATH=/home/edan/local_projects/phd_research
until grep -q saved collect_1_0.log && grep -q saved collect_0_3.log; do sleep 10; done
for T in m1:data_mode1_diff0.npz d3:data_mode0_diff3.npz; do
  NAME=${T%%:*}; FILE=${T#*:}
  for M in "new:UTILITY=curv PARENTS=loss MATURITY=0 PROTECT=rising" "old:MATURITY=50000" "lin:LINEAR=1"; do
    MN=${M%%:*}; OPTS=${M#*:}
    env OUT_GROW=1 WIDTH=600 N_SEEDS=1 N_STEPS=400000 $OPTS DATA_FILE=$FILE LABEL=fr_${NAME}_$MN /home/edan/miniconda3/envs/jax/bin/python run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU" >> log.txt
    env OUT_GROW=1 WIDTH=600 N_SEEDS=1 N_STEPS=400000 $OPTS SWITCH=200000 ALIGN=1 SWITCH_FILE=$FILE LABEL=sw_${NAME}_$MN /home/edan/miniconda3/envs/jax/bin/python run.py 2>&1 | grep -v -i "warn\|autotun\|pinned\|allocator\|bfc\|summary\|TF_GPU" >> log.txt
  done
done
echo batch18 done >> log.txt
