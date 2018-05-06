

traindirname = '../Audio Files/train';
testdirname = '../Audio Files/test';
codebook_size = 64;

run_clustering = false;
plot_training_data = false;
scatter_voice_data = false;
scatter_clustered_data = false;
liveDemo = false;
plot_voice_profile = true;

%% Grab Directories and Caluclate MFCCs (Training)

% For each valid directory, build array of MFCCs for this voice
% put in data cell array: {dirname : mfccs}

% Structure of dataMap: dirname: {filename, mfcc_data} as n x 2 matrix
trainedDataCells = cell(0);
aggregate = [];
subs = dir(traindirname);
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
                %MFCCs = (MFCCs-mean(MFCCs))./std(MFCCs);
                combined = [combined MFCCs];
            end
        end
        combined = combined'; %makek it an n X 13 array
        trainedDataCells(1,end+1) = {subs(i).name};
        trainedDataCells(2,end) = {combined};
        
        %add 64 cluster centroids
        [idx,c] = kmeans(combined, codebook_size); %c is a k-b-p matrix of centroids
        trainedDataCells(3,end) = {c};
                
        %add mean
        trainedDataCells(4,end) = {mean(combined)};
                
        %add normalized mean
        normCombined = (combined - mean(combined)) ./ std(combined);
        trainedDataCells(5,end) = {mean(normCombined)};
        
        %dataCell(3,end) = {mean(combined,2)};
        aggregate = [aggregate combined'];
    end
end
aggregate = aggregate'; %make it n X 13 array

%% Calculate MFCCs for Test Data

subs = dir(testdirname);
testDataCells = cell(0); % 3 by n dimensional cell array
    %n test samples
    %Name, mfcc matrix, 64 cluster centroids, mean, normalized mean
for i = 1:size(subs)
    name = subs(i).name;
    if name(1) == '.'
        continue
    end
    folderpath = fullfile(subs(i).folder, subs(i).name);    
    if isdir(folderpath)
        subfiles = dir(folderpath);
        %create cell array entry for this directory 
        for j = 1:size(subfiles) % Grab each file in subdirectory
            sub = subfiles(j);
            if sub.name(end) == 'v' %populate cell with filename
                filename = fullfile(sub.folder,sub.name);
                mffcs_for_filename;
                MFCCs = MFCCs'; %make it n X 13 array
                
                testDataCells(1,end+1) = {sub.name}; %add name
                testDataCells(2,end) = {MFCCs}; %add mfccs
               
                %add 64 cluster centroids
                [idx,c] = kmeans(MFCCs, codebook_size); %c is a k-b-p matrix of centroids
                testDataCells(3,end) = {c};
                
                %add mean
                testDataCells(4,end) = {mean(MFCCs)};
                
                %add normalized mean
                normMFCCs = (MFCCs - mean(MFCCs)) ./ std(MFCCs);
                testDataCells(5,end) = {mean(normMFCCs)};
                
            end
        end
    end
end


%% Run Kmeans Clustering on Aggregate Data (Find Number of Speakers via elbow)

if run_clustering
   minClusters = 1;
    maxClusters = 7;

    for k = minClusters:maxClusters

        [ids, centroids, sums] = kmeans(aggregate, k); % run kmeans
        sse = sum(sums.^ 2);
        sses(k - minClusters + 1) = sse;
    end 
end


%% Plot Elbow Curve

if run_clustering
    figure();
    subplot(2,1,1)
    plot(sses(1,1:maxClusters-minClusters+1));
    title('SSE with varying number of clusters')
    subplot(2,1,2)
    semilogy(sses(1,1:maxClusters-minClusters+1));
    title('SSE with varying number of clusters on Logarithmic Scale') 
end



%% Scatter pca data and plot pca centers

if plot_training_data
    pcaCoeffs = pca(aggregate);
    pcaData = aggregate * pcaCoeffs;
    pcaDataCell = cell(0);
    for i = 1:size(trainedDataCells,2) % run pca on each component
        if size(trainedDataCells{2,i},1) == size(pcaCoeffs,1)
            pcaDataCell{1,end+1} = trainedDataCells{1,i}; % copy name
            pcaDataCell{2,end} = trainedDataCells{2,i} * pcaCoeffs; %pca on data
            pcaDataCell{3,end} = trainedDataCells{3,i} * pcaCoeffs; %pca on centers
        else
        end

    end
end


%% Plot Data by voice

if scatter_voice_data
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
end



%% Plot data by cluster

if scatter_clustered_data
   figure();
    plot3(pcaData(ids==1,1),pcaData(ids==1,2),pcaData(ids==1,3),'o');
    hold on;
    for i = 2:max(ids)
        plot3(pcaData(ids==i,1),pcaData(ids==i,2),pcaData(ids==i,3),'o');
        disp(i)
    end
    title('Scatter of MFCC Vectors Colored by Clustering')

    legend 
end


%% Testing out classification method

if liveDemo
    figure();
    view(3);
    hold on;

    % getting colors for our speakers
    % number of speakers
    name_num = size(trainedDataCells(1,:));
    name_num = name_num(2);
    CM = jet(name_num); % # of unique colors for that # of speakers

     for i = 1:size(pcaDataCell,2) %Plot averages for each speaker
        c = pcaDataCell{3,i};
        plot3(c(:,1),c(:,2),c(:,3),'o', 'color', CM(i,:));
     end

    lgd = legend(trainedDataCells{1,:},'AutoUpdate','off');
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
    recording_mfccs = {};
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
end


%% Classify Recorded Data

if liveDemo
    recording_mfccs = audioCluster_new(recObj, false);
    disp('about to predict');

    [idx, c] = kmeans(recording_mfccs', codebook_size); %c is a k-b-p matrix of centroids
    recorded_profile = c; %64 X 13 vector that represents center of 64 clusters for voice profile
    
    for i=1:size(trainedDataCells, 1)
        %compute difference from voice profile to each profile in codebook    
        %distance(i) = calc_dissimilarity(recorded_profile, dataCell{3,i});
        [dists, ids] = pdist2(recorded_profile,trainedDataCells{3,i},'euclidean','Smallest',1);
        distance(i) = nanmean(dists);
        % compute euclidean distance to nearest mean
        %dist = norm(mfccs-map(labels{i}));
        %distances(i) = dist;
    end

    [min_val, min_ind] = min(distance);
    label = trainedDataCells{1,min_ind};
    disp(label);
    hold off;
    
end

%% Classify Test Data

classification_results = cell(0);
for i = 1:size(testDataCells,2) % Classify each test sample
    
    
    %Test against each voice profile
    for j = 1:size(trainedDataCells,2)
        [dists, ids] = pdist2(testDataCells{3,i}, trainedDataCells{3,j},'euclidean','Smallest',1);
        distance64(j) = nanmean(dists);
        distanceMean(j) = pdist2(testDataCells{4,i}, trainedDataCells{4,j},'euclidean','Smallest',1);%abs(nanmean(abs(testDataCells{4,i} - trainedDataCells{4,j})));
        distanceNormMean(j) = pdist2(testDataCells{5,i}, trainedDataCells{5,j},'euclidean','Smallest',1);%abs(nanmean(abs(testDataCells{5,i} - trainedDataCells{5,j})));
    end
    [min_val, min_ind] = min(distance64);
    sort(distance64);
    certainty = min_val / (min_val + distance64(2));
    classification_results(i,1) = {[testDataCells{1,i} ' - ' trainedDataCells{1,min_ind} '(' num2str(certainty) ')']};
    
    [min_val, min_ind] = min(distanceMean);
    sort(distanceMean);
    certainty = min_val / (min_val + distanceMean(2));
    classification_results(i,2) = {[testDataCells{1,i} ' - ' trainedDataCells{1,min_ind} '(' num2str(certainty) ')']};
    
    [min_val, min_ind] = min(distanceNormMean);
    sort(distanceNormMean);
    certainty = min_val / (min_val + distanceNormMean(2));
    classification_results(i,3) = {[testDataCells{1,i} ' - ' trainedDataCells{1,min_ind} '(' num2str(certainty) ')']};
end

classification_results(end+1,1) = {'64 Clusters'};
classification_results(end,2) = {'Mean'};
classification_results(end,3) = {'Normalized Mean'};


%% Plot a voice profile

if plot_voice_profile
    
    
    
end











