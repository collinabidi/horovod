#!/bin/bash
#SBATCH --job-name=horovod-test-2
#SBATCH --nodes=2 #number of nodes requested
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=mpi # mpi, gpu and smp are available in H2P
#SBATCH --output=results/horovod_shmem_2.txt
#SBATCH --mem=64g
#SBATCH --partition=ib
#SBATCH --time=0-20:00:00 # walltime in dd-hh:mm format
#SBATCH --qos=short # required if walltime is greater than 3 days

module purge #make sure the modules environment is sane
module load python/3.7.0 venv/wrap
module load gcc/8.2.0
module load openmpi/4.0.3

cd ~
workon henv
cd horovod/examples

oshrun -np 2 -x HOROVOD_MPI_THREADS_DISABLE=1 -x HOROVOD_SHMEM_THREADS_DISABLE=1 --mca mpi_cuda_support 0 \
	--mca pml ucx --mca osc ucx \
	--mca atomic ucx --mca orte_base_help_aggregate 0 \
	--mca btl ^vader,tcp,openib,uct python3 pytorch_basic.py --epochs 1 --no-cuda
