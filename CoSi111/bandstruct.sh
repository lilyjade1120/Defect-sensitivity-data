#!/bin/bash
#SBATCH -p el8 
#SBATCH --gres=gpu:6
#SBATCH -n 60 
#SBATCH -c 1 
#SBATCH --ntasks-per-node=6
#SBATCH -t 6:00:00
#SBATCH --hint=nomultithread 
#SBATCH -J bands
#SBATCH --mail-use=joycel@rpi.edu
#SBATCH --mail-type=END

module purge
module load openmpi-gpu mkl gsl libxc hdf5
module load spectrum-mpi

#Run job:
export OMPI_MCA_opal_cuda_support=true

mpirun -gpu jdftx_gpu -mi bandstruct.in
