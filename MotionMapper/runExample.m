% parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')))
clock
%PLACE PATH TO FOLDER CONTAINING BODY POSITION HERE
filePath = FILE_PATH

%add utilities folder to path
addpath(genpath('./utilities/'));
addpath(genpath('./PCA/'));
addpath(genpath('./segmentation_alignment/'));
addpath(genpath('./t_sne/'));
addpath(genpath('./wavelet/'));

%find all avi files in 'filePath'
% make sure to have MAT in front of .mat filename
imageFiles = findAllImagesInFolders(filePath,'.mat', 'MAT');
L = length(imageFiles);
numZeros = ceil(log10(L+1e-10));

%define any desired parameter changes here
% parameters.samplingFreq = 50;
% parameters.trainingSetSize = 35000; % previously 1000
% parameters.numPeriods = 25;
% parameters.omega0 = 5;
% parameters.minF = 1;
% parameters.maxF = 25;
% parameters.maxNeighbors = 200; % MUST BE LESS THAN SAMPLE; previously 30
% parameters.perplexity = 40; % LESS THAN BERMAN'S 32; previously 28

parameters.samplingFreq = SAMPLING_FREQ;
parameters.trainingSetSize = TRAINING_SET_SIZE;
parameters.numPeriods = NUM_PERIODS;
parameters.omega0 = OMEGA_0;
parameters.minF = MIN_F;
parameters.maxF = MAX_F;
parameters.maxNeighbors = MAX_NEIGHBORS;
parameters.perplexity = PERPLEXITY;
parameters.pcaModes = 22; % CHANGE THIS LATER; ONLY THE CASE FOR RAT

numCoresString = getenv('SLURM_CPUS_PER_TASK');

if isempty(numCoresString)
    parameters.numProcessors=2;  % just use a default value outside SLURM
else
    parameters.numProcessors=str2double(numCoresString);
end
fprintf(1, strcat("Number of Processor Used: ", numCoresString, '\n'));

%initialize parameters
parameters = setRunParameters(parameters);

projectionsDirectory = [filePath];
if ~exist(projectionsDirectory,'dir')
    mkdir(projectionsDirectory);
end
%% Use subsampled t-SNE to find training set; IMPORTANCE SAMPLING
fprintf(1,'Finding Training Set\n');
[trainingSetData,trainingSetAmps,projectionFiles] = runEmbeddingSubSampling(projectionsDirectory,parameters);

%% Run t-SNE on training set
fprintf(1,'Finding t-SNE Embedding for the Training Set\n');
[trainingEmbedding,betas,P,errors] = run_tSne(trainingSetData,parameters);

%% Find Embeddings for each file
fprintf(1,'Finding t-SNE Embedding for each file\n');
embeddingValues = cell(L,1);
for i=1:L
    fprintf(1,'\t Finding Embbeddings for File #%4i out of %4i\n',i,L);
    load(projectionFiles{i},'projections');
    projections = projections(:,1:parameters.pcaModes);
    [embeddingValues{i},~] = findEmbeddings(projections,trainingSetData,trainingEmbedding,parameters);
    % save corresponding files data
    embed_values_i = embeddingValues{i};
    dir_part = strsplit(string(imageFiles{i}), '/');
    num_arg = dir_part.length-1;
    save(strcat(join(dir_part(1:num_arg),'/'),'/EMBED.mat'), 'embed_values_i');
    clear projections
    clear embed_values_i
end

delete(gcp);
clock
close_parpool

