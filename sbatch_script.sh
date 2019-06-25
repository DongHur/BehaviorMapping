#!/bin/bash
#SBATCH -n 5 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=5
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

module load matlab
matlab -nodisplay -nodesktop -nosplash <<EOF
distcomp.feature( 'LocalUseMpiexec', false );
pc = parcluster('local');
pc.JobStorageLocation = ['/scratch/',getenv('USER'),'/46630138'];
p = parpool(pc, 5);
p.IdleTimeout = Inf;

cd ~/BehaviorMapping/MotionMapper
compile_mex_files

addpath(genpath('~/BehaviorMapping'));
cd ~/BehaviorMapping/MotionMapper
runExample;
save('../results/workspace_data.mat', 'embeddingValues');  

poolobj = gcp('nocreate');
delete(poolobj);

exit;
EOF

# Clean local work directory
rm -rf /scratch/$USER/46630138