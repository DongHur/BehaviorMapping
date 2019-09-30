function [templates,xx,density,sigma,lengths,L,vals2] = ...
    returnTemplates(yData,signalData,minTemplateLength,kdNeighbors,plotsOn)
%returnTemplates is used by findTemplatesFromData.m to group wavelets
%mapped to the same region of t-SNE embedded space

    if nargin < 3 || isempty(minTemplateLength)
        minTemplateLength = 10;
    end
    
    if nargin < 4 || isempty(kdNeighbors)
        kdNeighbors = 10;
    end
    
    if nargin < 5 || isempty(plotsOn)
        plotsOn = true;
    end


    maxY = ceil(max(abs(yData(:)))) + 1;
    d = length(signalData(1,:));
    
    NS = createns(yData);
    [~,D] = knnsearch(NS,yData,'K',kdNeighbors+1);
    
    sigma = median(D(:,kdNeighbors+1));

   [xx,density] = findPointDensity(yData,sigma,501,[-maxY maxY]);


    L = watershed(-density,8);
    vals = round((yData + max(xx))*length(xx)/(2*max(xx)));

    N = length(D(:,1));
    watershedValues = zeros(N,1);
    fprintf(1, strcat("Test1: ", num2str(vals(1,2)), '\n'));
    fprintf(1, strcat("Test2: ", num2str(N), '\n'));
    for i=1:N
        fprintf(1, strcat("Test3: ", num2str(i), '\n'));
        watershedValues(i) = diag(L(vals(i,2),vals(i,1)));
    end
    
    maxL = max(L(:));
    templates = cell(maxL,1);
    for i=1:maxL
        templates{i} = signalData(watershedValues==i,:);
    end
    lengths = returnCellLengths(templates) / d;

    idx = find(lengths >= minTemplateLength);
    vals2 = zeros(size(watershedValues));
    for i=1:length(idx)
        vals2(watershedValues == idx(i)) = i;
    end
        
    templates = templates(lengths >= minTemplateLength);
    lengths = lengths(lengths >= minTemplateLength);

    
    
    if plotsOn
        imagesc(xx,xx,density);
        set(gca,'ydir','normal');
        axis equal tight;
        hold on
        [ii,jj] = find(L==0);
        plot(xx(jj),xx(ii),'k.')
    end