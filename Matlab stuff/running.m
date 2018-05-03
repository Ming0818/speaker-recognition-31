

dirname = '../Audio Files/test';


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
                % subtracting mean and dividing standard dev 
                MFCCs = (MFCCs-mean(MFCCs))./std(MFCCs);
                combined = [combined MFCCs];
            end
        end
        dataCell(1,end+1) = {subs(i).name};
        dataCell(2,end) = {combined};
        dataCell(3,end) = {mean(combined,2)};
        aggregate = [aggregate combined];
    end
end


%% Run Kmeans Clustering on Aggregate Data

minClusters = 2;
maxClusters = 7;

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

%% Testing out classification method

figure();
hold on;

% getting colors for our speakers
% number of speakers
name_num = size(dataCell(1,:));
name_num = name_num(2);
CM = jet(name_num); % # of unique colors for that # of speakers

 for i = 1:size(pcaDataCell,2) %Plot averages for each speaker
    c = pcaDataCell{3,i};
    plot3(c(:,1),c(:,2),c(:,3),'o', 'color', CM(i,:));
 end

lgd = legend(dataCell{1,:},'AutoUpdate','off');
lgd.FontSize = 14;
title('Scatter of MFCC Vectors Colored By Speaker')


recObj = audiorecorder(44100, 16, 1); % default is 9000 sample rate
% if we want to set functions to tell us when we're starting and stopping
% the recording
% recObj.StartFcn = 'disp(''Start speaking.'')';
% recObj.StopFcn = 'disp(''End of recording.'')';
% timerFcn doesn't seem to work, so instead use the loop below

% loop that plots MFCCs for every 1 second recording of data, so you can in
% almost real-time watch your voice cluster 

% will loop until you press 'q' 
total = {};
finish=false;
set(gcf,'CurrentCharacter','@'); % set to a dummy character
disp('Start recording');
while ~finish
  hold on;
 % total = vertcat(total, t);
  % check for keys
  record(recObj);
  pause(1);
  pause(recObj);
  t = audioCluster_new(recObj, false);
  k=get(gcf,'CurrentCharacter');
  if k~='@' % has it changed from the dummy character?
    set(gcf,'CurrentCharacter','@'); % reset the character
    % now process the key as required
    if k=='q', finish=true; end
  end
  resume(recObj);
end
stop(recObj);
disp('End recording');
total = audioCluster_new(recObj, false);
disp('about to predict');


% hashmap of labels:means (e.g. name:[mean_of_mfccs])
map = containers.Map(dataCell(1,:), dataCell(3,:));

label = spkr_classify(total, map, true);
disp(label);


hold off;

