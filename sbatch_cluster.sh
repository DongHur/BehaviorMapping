#!/bin/bash
#SBATCH -c 2 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 45 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=5G # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -o logs/myoutput_%j.out #Standard out goes to this file
#SBATCH -e logs/myerr_%j.err # Standard err goes to this filehostname
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhur@college.harvard.edu

# if you ask for a certain number of cores, the memory size also has to match a minimum amount
# 8 cores need at least 10G

module load Anaconda/5.0.1-fasrc02
conda create -n behavior --clone $PYTHON_HOME
source activate behavior
python cluster.py
source deactivate

