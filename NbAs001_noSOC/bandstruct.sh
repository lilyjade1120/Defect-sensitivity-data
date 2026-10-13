#!/bin/bash
#SBATCH -p el8 
#SBATCH -t 06:00:00 
#SBATCH -n 24 
#SBATCH -c 1 
#SBATCH --ntasks-per-node=6 
#SBATCH --gres=gpu:6 
#SBATCH --hint=nomultithread 
#SBATCH -J 001NbAs
#SBATCH --mail-use=joycel@rpi.edu
#SBATCH --mail-type=END
#SBATCH -o bandstruct.out

module load spectrum-mpi

#Run job:
export JDFTX_CACHE_SIZE=30000
export OMPI_MCA_opal_cuda_support=true

mpirun -gpu /gpfs/u/home/BEAS/BEASjcll/barn-shared/dcs/JDFTx/build-managed/jdftx_gpu -i bandstruct.in
