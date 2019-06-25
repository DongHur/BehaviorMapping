#!/bin/bash
#SBATCH -n 5 # Number of cores requested
#SBATCH --ntasks-per-node=5
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 30 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=1000 # Memory per cpu in MB (see also --mem-per-cpu) 10GB
#SBATCH --open-mode=append
#SBATCH -o results/myoutput_%j.out # Standard out goes to this file
#SBATCH -e results/myerr_%j.err # Standard err goes to this filehostname

module load matlab
matlab -nodisplay -nodesktop -nosplash -wait -log -r "run('MotionMapper/runExample.m'); save('../results/workspace_data.mat', 'embeddingValues');  exit;"  | tail -n +11