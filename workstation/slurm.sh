#!/bin/bash

#SBATCH --partition short
#SBATCH --mem-per-cpu 8G
#SBATCH --time 2:00:00
#SBATCH --job-name ABP-data
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py -f noshear.txt --PD -l1 0.25 -u1 4 -l2 0.125 -u2 2 -n 16
