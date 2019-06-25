#!/bin/bash
#SBATCH -J pfor
#SBATCH -c 5 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 30 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=1000 # Memory per cpu in MB (see also --mem-per-cpu) 10GB
#SBATCH --open-mode=append
#SBATCH -o results/myoutput_46630138.out # Standard out goes to this file
#SBATCH -e results/myerr_46630138.err # Standard err goes to this filehostname

# Create a local work directory
mkdir -p /scratch/$USER/46630138

# Create results folder
mkdir -p ~/BehaviorMapping/results/

module load matlab/R2018b-fasrc01
srun -c $SLURM_CPUS_PER_TASK matlab -nodisplay -nodesktop -nosplash <<EOF
distcomp.feature( 'LocalUseMpiexec', false );
parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')));

cd ~/BehaviorMapping/MotionMapper
compile_mex_files

addpath(genpath('~/BehaviorMapping'));
cd ~/BehaviorMapping/MotionMapper
runExample;
save('../results/workspace_data.mat', 'embeddingValues');  

delete(gcp);
exit;
EOF

# Clean local work directory
rm -rf /scratch/$USER/46630138