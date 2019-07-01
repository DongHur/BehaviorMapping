#!/bin/bash
#SBATCH -J pfor
#SBATCH -c 8 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 30 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=10G # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -o logs/myoutput_%j.out #Standard out goes to this file
#SBATCH -e logs/myerr_%j.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhur@college.harvard.edu

module load matlab/R2018b-fasrc01
srun -c $SLURM_CPUS_PER_TASK matlab -nodisplay -nodesktop -nosplash -r "run('MotionMapper/runExample.m'); exit;"
