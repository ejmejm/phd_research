#!/bin/bash
cd /home/edan/local_projects/phd_research/phd/sandbox/one_off_experiments/ateroids_model
export PYTHONPATH=/home/edan/local_projects/phd_research IPYTHONDIR=/tmp/claude-1000/-home-edan-local-projects-phd-research/e6980810-12af-4154-834e-bbae7bf8032f/scratchpad/ipython XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/edan/miniconda3/envs/jax/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 06_scaling_unit_search.ipynb > scratch/exec06.log 2>&1
echo exec06 done >> scratch/exec06.log
