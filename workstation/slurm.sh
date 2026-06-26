#!/bin/bash

#SBATCH --partition medium
#SBATCH --mem-per-cpu 8G
#SBATCH --time 4:00:00
#SBATCH --job-name ABP
#SBATCH --error error-%j.txt
#
#######################################

source .venv/bin/activate
python abp_simulation.py -f pd500.txt --PD -l1 0.25 -u1 4 -l2 0.125 -u2 2 -n 16 -G 1 -N 500