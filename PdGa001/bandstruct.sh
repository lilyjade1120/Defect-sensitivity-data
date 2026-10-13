#!/bin/bash
#SBATCH -p cpu
#SBATCH -n 160
#SBATCH -c 1
##SBATCH --gres=gpu:6
#SBATCH --hint=nomultithread 
#SBATCH -J bands
#SBATCH -o bandstruct.out

module purge
module load openmpi-gpu mkl gsl libxc hdf5
module use /home/shankar/modules
module load jdftx

mpirun jdftx -i bandstruct.in
