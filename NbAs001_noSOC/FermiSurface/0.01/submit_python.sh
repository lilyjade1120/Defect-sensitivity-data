#!/bin/bash
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 2-0:00:00
#SBATCH -J FS 
#SBATCH -o python.out

module use /home/shankar/modules
module load venv/qimpy

python3 FermiSurface2D.py 0.01 
