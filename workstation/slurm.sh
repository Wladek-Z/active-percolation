#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 8G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP-data
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_analysis.py -f pd3.txt --PD --data -l 0.25 -u 4 -n 16
