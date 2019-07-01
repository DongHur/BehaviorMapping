#!/bin/bash
#SBATCH -J pfor
#SBATCH -c 8 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 30 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=10G # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -o results/myoutput_%j.out # Standard out goes to this file
#SBATCH -e results/myerr_%j.err # Standard err goes to this filehostname


module load Anaconda3/5.0.1-fasrc02
python run.py

module load matlab/R2018b-fasrc01
srun -c $SLURM_CPUS_PER_TASK matlab -nodisplay -nodesktop -nosplash -r "run('MotionMapper/runExample.m');  exit;"
