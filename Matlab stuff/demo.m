%% Speaker Identification Demo Using Pitch and MFCC

%% Data Set

dataDir = '/Users/lindsayhexter/Documents/GitHub/speaker-recognition/Audio Files/test/';
%%
% Create an |audioexample.Datastore| object to easily manage this database
% for training. The datastore allows you to collect necessary files of a
% file format and read them.
ads = audioexample.Datastore(dataDir, 'IncludeSubfolders', true,...
    'FileExtensions', {'.wav', '.m4a'}, 'ReadMethod','File',...
    'LabelSource','foldernames')

%%
% The |splitEachLabel| method of |audioexample.Datastore| splits the
% datastore into two or more datastores. The resulting datastores have the
% specified proportion of the audio files from each label. In this example,
% the datastore is split into two parts. 80% of the data for each label is
% used for training, and the remaining 20% is used for testing. The
% |countEachLabel| method of |audioexample.Datastore| is used to count the
% number of audio files per label. In this example, the label identifies
% the speaker.
[trainDatastore, testDatastore]  = splitEachLabel(ads,0.80);

%%
% Display the datastore and the number of speakers, in the train datastore.
trainDatastore
trainDatastoreCount = countEachLabel(trainDatastore)

%%
% Display the datastore and the number of speakers, in the test datastore.
testDatastore
testDatastoreCount = countEachLabel(testDatastore)

%%
% To preview the content of your datastore, read a sample file and play it
% using your default audio device.
% [sampleTrain, info] = read(trainDatastore);
% sound(sampleTrain,info.SampleRate)

%%
% Reading from the train datastore pushes the read pointer so that you can
% iterate through the database. Reset the train datastore to return the
% read pointer to the start for the following feature extraction.
% reset(trainDatastore); 

%% Feature Extraction
% Pitch and MFCC features are extracted from each frame using
% HelperComputePitchAndMFCC |<matlab:edit('HelperComputePitchAndMFCC')
% HelperComputePitchAndMFCC>| which performs the following actions on the
% data read from each audio file:
%
% # Collect the samples into frames of 30 ms with an overlap of 75%.
% # For each frame, use
% |<matlab:edit('audiopluginexample.SpeechPitchDetector.isVoicedSpeech')
% audiopluginexample.SpeechPitchDetector.isVoicedSpeech>| to decide whether
% the samples correspond to a voiced speech segment.
% # Compute the pitch and 13 MFCCs (with the first MFCC coefficient
% replaced by log-energy of the audio signal) for the entire file.
% # Keep the pitch and MFCC information pertaining to the voiced frames
% only.
% # Get the directory name for the file. This corresponds to the name of
% the speaker and will be used as a label for training the classifier.
% 
% |HelperComputePitchAndMFCC| returns a table containing the filename,
% pitch, MFCCs, and label (speaker name) as columns for each 30 ms frame.
lenDataTrain = length(trainDatastore.Files);
features = cell(lenDataTrain,1);
for i = 1:lenDataTrain
    [dataTrain, infoTrain] = read(trainDatastore); 
    % cutting off data - before ran for like 20min and didn't stop b/c of billion nested
    % for-loop
    len_ = size(dataTrain);
    len_ = len_(1);
    dataTrain = dataTrain(1:(int16(len_)*1),:);
    
    % checking if the data gave more than one frame - cut it off
    len = size(dataTrain);
    if len(2) > 1
        dataTrain = dataTrain(:,1);
        infoTrain = infoTrain(:,1);
    end
    features{i} = HelperComputePitchAndMFCC(dataTrain,infoTrain);

end
features = vertcat(features{:});
features = rmmissing(features);
head(features)   % Display the first few rows

%% 
% Notice that the pitch and MFCC are not on the same scale. This will bias
% the classifier. Normalize the features by subtracting the mean and
% dividing the standard deviation of each column.
featureVectors = features{:,2:15};

m = mean(featureVectors);
s = std(featureVectors);
features{:,2:15} = (featureVectors-m)./s;
head(features)   % Display the first few rows


%% Training a Classifier
% Now that you have collected features for all ten speakers, you can train
% a classifier based on them. In this example, you use a K-nearest neighbor
% classifier defined in |<matlab:edit('HelperTrainKNNClassifier')
% HelperTrainKNNClassifier>|. K-nearest neighbor is a classification
% technique naturally suited for multi-class classification. The
% hyperparameters for the nearest neighbor classifier include the number of
% nearest neighbors, the distance metric used to compute distance to the
% neighbors, and the weight of the distance metric. The hyperparameters are
% selected to optimize validation accuracy and performance on the test set.
% In this example, the number of neighbors is set to 5 and the metric for
% distance chosen is squared-inverse weighted Euclidean distance. For more
% information about the classifier, refer to
% |<matlab:web(fullfile(docroot,'stats/fitcknn.html'),'-helpbrowser')
% fitcknn>|.

[trainedClassifier, validationAccuracy, confMatrix] = ...
    HelperTrainKNNClassifier(features);
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
heatmap(trainedClassifier.ClassNames, trainedClassifier.ClassNames, ...
    confMatrix);
title('Confusion Matrix');
%% 
% plotting with pca to visualize our feature vectors 
figure();
names = unique(table2array(features(:,16)));
len = size(names);

CM = jet(len(1));
indices = (1:len(1));
map = containers.Map(names, indices);

for i=1:size(features)
    name = char(features{i, 16});
    
    color = CM(map(name),:);
    temp = features(i,3:15);
    temp = temp{:,:};
    pcaCoeffs = pca(temp');
    pcaTest = temp*pcaCoeffs;
    % plotting the pca
    plot3(pcaTest(:,1), pcaTest(:, 2), pcaTest(:, 3), 'o', 'color',color, 'MarkerFaceColor', color);
    
    hold on;
end

l_list = {};
for i=1:size(names)
    name= names{i};
    disp(names{i});
    l_list = [l_list, ['\color[rgb]{' num2str(CM(map(name), 1)) ',' num2str(CM(map(name), 2)) ',' num2str(CM(map(name), 3))  '} ' name]];
    
end 

lgd = legend(l_list,'AutoUpdate','off');
lgd.FontSize = 14;


recObj = audiorecorder; % default is 9000 sample rate
% if we want to set functions to tell us when we're starting and stopping
% the recording
% recObj.StartFcn = 'disp(''Start speaking.'')';
% recObj.StopFcn = 'disp(''End of recording.'')';
% timerFcn doesn't seem to work, so instead use the loop below
set(recObj,'TimerPeriod',1)%,'TimerFcn',{@audioCluster, recObj});

% loop that plots MFCCs for every 1 second recording of data, so you can in
% almost real-time watch your voice cluster 

% will loop until you press 'q' 
total = {};
finish=false;
set(gcf,'CurrentCharacter','@'); % set to a dummy character
while ~finish
  t = audioCluster(recObj);
  total = vertcat(total, t);
  % check for keys
  k=get(gcf,'CurrentCharacter');
  if k~='@' % has it changed from the dummy character?
    set(gcf,'CurrentCharacter','@'); % reset the character
    % now process the key as required
    if k=='q', finish=true; end
  end
end
disp('about to predict');
total = arrayfun(@(col) vertcat(total{:, col}), 1:size(total, 2), 'UniformOutput', false);
total = total{1};
predictedLabels = string(predict(trainedClassifier,total)); % Predict
totalVals = size(predictedLabels,1);

[predictedLabel, freq] = mode(predictedLabels); % Find most frequently predicted label
match = freq/totalVals*100;

disp(predictedLabel);