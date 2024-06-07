#!/bin/bash

#SBATCH --account=rsteven1_gpu
#SBATCH --job-name=healing_1
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --time=30:00:00
#SBATCH --mem=50G
#SBATCH --export=ALL

python /home/agoyal19/My_Work/Healing_Model/Lesion_Healing.py
