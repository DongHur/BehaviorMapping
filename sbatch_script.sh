#!/bin/bash
#SBATCH -n 5 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 30 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=1000 # Memory per cpu in MB (see also --mem-per-cpu) 10GB
#SBATCH --open-mode=append
#SBATCH -o myoutput_%j.out # Standard out goes to this file
#SBATCH -e myerr_%j.err # Standard err goes to this filehostname

module load matlab
matlab -nodisplay -nodesktop -nosplash -nojvm -wait -log -r "run('script_test.m'); save('workspace_data.mat', 'embeddingValues');  exit;"  | tail -n +11