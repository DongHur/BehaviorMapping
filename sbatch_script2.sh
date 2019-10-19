#!/bin/bash
#SBATCH -J pfor
#SBATCH -c 30 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 24:0:0 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=70G # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -o logs/myoutput_%j.out #Standard out goes to this file
#SBATCH -e logs/myerr_%j.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhur@college.harvard.edu

# if you ask for a certain number of cores, the memory size also has to match a minimum amount
# 8 cores need at least 10G

module load matlab/R2018b-fasrc01
FILE_PATH='../data/rat_data'
SAMPLING_FREQ=30
TRAINING_SET_SIZE=50000
NUM_PERIODS=25
OMEGA_0=5
MIN_F=1
MAX_F=25
MAX_NEIGHBORS=200
PERPLEXITY=40

srun -c $SLURM_CPUS_PER_TASK matlab -nodisplay -nodesktop -nosplash -r \
"FILE_PATH='$FILE_PATH'; \
SAMPLING_FREQ=$SAMPLING_FREQ; \
TRAINING_SET_SIZE=$TRAINING_SET_SIZE; \
NUM_PERIODS=$NUM_PERIODS; \
OMEGA_0=$OMEGA_0; \
MIN_F=$MIN_F; \
MAX_F=$MAX_F; \
MAX_NEIGHBORS=$MAX_NEIGHBORS; \
PERPLEXITY=$PERPLEXITY; \
run('MotionMapper/runExample.m'); \
exit;"
