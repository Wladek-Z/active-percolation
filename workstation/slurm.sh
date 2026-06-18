#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 8G
#SBATCH --time 6:00:00
#SBATCH --job-name test
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_analysis.py -f pd_test.txt --PD --data -l 0.5 -u 5 -n 10
