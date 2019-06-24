#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10 
#SBATCH -p serial_requeue
#SBATCH --mem=500
#SBATCH -o myoutput_%j.out
#SBATCH -e myerr_%j.err

module load matlab
matlab -nodisplay -nodesktop -nosplash -nojvm -wait -log -r "run('script_test_SAMPLE.m'); save('workspace_data.mat', 'embeddingValues'); exit;"  | tail -n +11