

dirname = '../Audio Files';


%% Grab Directories and Caluclate MFCCs

% For each valid directory, build array of MFCCs for this voice
% put in data cell array: {dirname : mfccs}

% Structure of dataMap: dirname: {filename, mfcc_data} as n x 2 matrix
dataCell = cell(0);
aggregate = [];
subs = dir(dirname);
for i = 1:size(subs)
    name = subs(i).name;
    if name(1) == '.'
        continue
    end
    folderpath = fullfile(subs(i).folder, subs(i).name);
    combined = [];
    
    if isdir(folderpath) %Grab files and add to allfiles
        subfiles = dir(folderpath);
        %create cell array entry for this directory 
        for j = 1:size(subfiles)
            sub = subfiles(j);
            if sub.name(end) == 'v' %populate cell with filename
                filename = fullfile(sub.folder,sub.name);
                mffcs_for_filename;
                combined = [combined MFCCs];
            end
        end
        dataCell(1,end+1) = {name};
        dataCell(2,end) = {combined};
        dataCell(3,end) = {mean(combined,2)};
        aggregate = [aggregate combined];
    end
end


%% Run Kmeans Clustering on Aggregate Data

minClusters = 8;
maxClusters = 20;

for k = minClusters:maxClusters
   
    [ids, centroids, sums] = kmeans(aggregate', k); % run kmeans
    sse = sum(sums.^ 2);
    sses(k - minClusters + 1) = sse;
end


%% Plot Elbow Curve

figure();
subplot(2,1,1)
plot(sses(1,1:maxClusters-minClusters+1));
title('SSE with varying number of clusters')
subplot(2,1,2)
semilogy(sses(1,1:maxClusters-minClusters+1));
title('SSE with varying number of clusters on Logarithmic Scale')


%% Scatter pca data and plot pca centers

pcaCoeffs = pca(aggregate');
pcaData = aggregate' * pcaCoeffs;
pcaDataCell = cell(0);
for i = 1:size(dataCell,2) % run pca on each component
    if size(dataCell{2,i},1) == size(pcaCoeffs,1)
        pcaDataCell{1,end+1} = dataCell{1,i}; % copy name
        pcaDataCell{2,end} = dataCell{2,i}' * pcaCoeffs; %pca on data
        pcaDataCell{3,end} = dataCell{3,i}' * pcaCoeffs; %pca on centers
    else
    end
    
end

%% Plot Data by voice
figure();
c = pcaDataCell{2,1};
plot3(c(:,1),c(:,2),c(:,3),'o');
hold on;
for i = 2:size(pcaDataCell,2) %plot data points
    c = pcaDataCell{2,i};
    plot3(c(:,1),c(:,2),c(:,3),'o');
end

 for i = 1:size(pcaDataCell,2) %Plot averages for each speaker
    c = pcaDataCell{3,i};
    plot3(c(:,1),c(:,2),c(:,3),'o');
end
title('Scatter of MFCC Vectors Colored By Speaker')
hold off;


%% Plot data by cluster
figure();
plot3(pcaData(ids==1,1),pcaData(ids==1,2),pcaData(ids==1,3),'o');
hold on;
for i = 2:max(ids)
    plot3(pcaData(ids==i,1),pcaData(ids==i,2),pcaData(ids==i,3),'o');
    disp(i)
end
title('Scatter of MFCC Vectors Colored by Clustering')

legend

%% Plot data in 3d corresponding to voices





