%% Add Proper Files
addpath(genpath('./utilities/'));
addpath(genpath('./PCA/'));
addpath(genpath('./segmentation_alignment/'));
addpath(genpath('./t_sne/'));
addpath(genpath('./wavelet/'));
%% Load Data
path = '../data_result/data_final/**/EMBED.mat';
%get data for each file
EMBED_filepath = dir(path);
file_info = struct2cell(EMBED_filepath);
L = length(file_info(2,:));
embeddingValues = cell(L,1);
for i=1:L
    filepath = strcat(file_info(2,i),'/',file_info(1,i));
    embed_data = load(filepath{1});
    embeddingValues{i} = embed_data.embed_values_i;
end

%% Make density plots
%load data
%add utilities folder to path

L = length(embeddingValues); 
maxVal = max(max(abs(combineCells(embeddingValues))));
maxVal = round(maxVal * 1.1);

sigma = maxVal / 40;
numPoints = 501;
rangeVals = [-maxVal maxVal];

[xx,density] = findPointDensity(combineCells(embeddingValues),sigma,numPoints,rangeVals);

densities = zeros(numPoints,numPoints,L);
for i=1:L
    [~,densities(:,:,i)] = findPointDensity(embeddingValues{i},sigma,numPoints,rangeVals);
end

%plot total mapping figure
figure
maxDensity = max(density(:));
imagesc(xx,xx,density)
axis equal tight off xy
caxis([0 maxDensity * .8])
colormap(jet)
colorbar

%plot individual mapping figure
h = figure;
% N = ceil(sqrt(L));
% M = ceil(L/N);
maxDensity = max(densities(:));
for i=1:L
%     subplot(M,N,i)
    imagesc(xx,xx,densities(:,:,i))
    axis equal tight off xy
    caxis([0 maxDensity * .8])
    colormap(jet)
    title(['Data Set #' num2str(i)],'fontsize',12,'fontweight','bold');
    fig_filepath = char(file_info(2,i));
    saveas(h, fullfile(fig_filepath,'indiv_mat'), 'fig');
end

%% Modify tSNE points
total_tsne = cat(1,embeddingValues{:});
bad_frames = [];
% for i=1:length(total_tsne)
%     if total_tsne(i,1) < -60
%         % take out black screen points or zero body movement
%         bad_frames = [bad_frames, i];
%     elseif (total_tsne(i,1) < -10) && (total_tsne(i,2) < -10)
%         % take out body frames with wrong body point
%         bad_frames = [bad_frames, i];
%     elseif (total_tsne(i,1) < 28) && (total_tsne(i,2) < -50)
%         % take out body frames with wrong body point
%         bad_frames = [bad_frames, i];
%     end
% end
total_tsne(bad_frames,:) = [];
disp("Finished Filtering Data");
% setup density plot parameter
L = length(embeddingValues); 
maxVal = max(max(abs(total_tsne)));
maxVal = round(maxVal * 1.1);

sigma = maxVal / 40;
numPoints = 501;
rangeVals = [-maxVal maxVal];

[xx,density] = findPointDensity(total_tsne,sigma,numPoints,rangeVals);

% plot total mapping figure
figure
maxDensity = max(density(:));
imagesc(xx,xx,density)
axis tight on xy
%******** no log scale
caxis([0 maxDensity * .8])
colormap(jet)
cb = colorbar();
grid on
grid minor
ax = gca
ax.LineWidth = 2
ax.GridLineStyle = '-'
ax.GridColor = 'w'
ax.GridAlpha = 0.3 % maximum line opacity
% Watershed Plot
hold on
L = watershed(-density,8);
[ii,jj] = find(L==0);
disp(L)
plot(xx(jj),xx(ii),'k.')

%% Computes Just Watershed Region
L = watershed(-density,8);
rgb = label2rgb(L,'jet',[.5 .5 .5]);
figure
imshow(rgb,'InitialMagnification','fit')
