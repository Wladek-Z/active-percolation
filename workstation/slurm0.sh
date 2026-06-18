#!/bin/bash

#SBATCH --job-name=ABP-data
#SBATCH --partition=long
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#
#######################################

source .venv/bin/activate
python test.py
