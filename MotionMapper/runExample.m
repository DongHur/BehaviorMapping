parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')))
%%example script that will run the code for a set of .avi files that are
%%found in filePath
clock
%Place path to folder containing example .avi files here
%PLACE PATH TO FOLDER CONTAINING BODY POSITION HERE
filePath = 'data_mat/bpdata';

%add utilities folder to path
addpath(genpath('./utilities/'));
addpath(genpath('./PCA/'));
addpath(genpath('./segmentation_alignment/'));
addpath(genpath('./t_sne/'));
addpath(genpath('./wavelet/'));

%find all avi files in 'filePath'
imageFiles = findAllImagesInFolders(filePath,'.mat');
L = length(imageFiles);
numZeros = ceil(log10(L+1e-10));

%define any desired parameter changes here
parameters.samplingFreq = 50;
parameters.trainingSetSize = 1000;
parameters.numPeriods = 25;
parameters.omega0 = 5;
parameters.minF = 1;
parameters.maxF = 25;
parameters.maxNeighbors = 30; % MUST BE LESS THAN SAMPLE
parameters.perplexity = 28; % LESS THAN BERMAN'S 32

numCoresString=getenv('SLURM_NTASKS_PER_NODE');
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
    clear projections
end

delete(gcp);
clock
close_parpool

