%% Speaker Identification Using Pitch and MFCC
% This example demonstrates a machine learning approach to identify people
% based on features extracted from recorded speech. The features used to
% train the classifier are: pitch of the voiced segments of the speech, and
% the Mel Frequency Cepstrum Coefficients (MFCC). This is a closed-set
% speaker identification - the audio of the speaker under test is compared
% against all the available speaker models (a finite set) and the closest
% match is returned.
%
% This example requires Statistics and Machine Learning Toolbox.

%   Copyright 2017-2018 The MathWorks, Inc.

%% Introduction
% The approach used in this example for speaker identification is shown in
% the diagram.
%
% <<SpeakerID01.png>>
%
% Pitch and Mel Frequency Cepstrum Coefficients (MFCC) are extracted from
% speech signals recorded for 10 speakers. These features are used to train
% a K-Nearest Neighbor (KNN) classifier. Then, new speech signals that need
% to be classified go through the same feature extraction. The trained KNN
% classifier predicts which one of the ten speakers is the closest match.

%% Features Used for Classification
% This section discusses pitch and Mel Frequency Cepstrum Coefficients
% (MFCCs), the two features that are used to classify speakers.
%
% *Pitch*
%
% Speech can be broadly categorized as _voiced_ and _unvoiced_. In the case
% of voiced speech, air from the lungs is modulated by vocal cords and
% results in a quasi-periodic excitation. The resulting sound is dominated
% by a relatively low-frequency oscillation, referred to as _pitch_. In the
% case of unvoiced speech, air from the lungs passes through a constriction
% in the vocal tract and becomes a turbulent, noise-like excitation. In the
% source-filter model of speech, the excitation is referred to as the
% source, and the vocal tract is referred to as the filter. Characterizing
% the source is an important part of characterizing the speech system.
%
% As an example of voiced and unvoiced speech, consider a time-domain
% representation of the word "two" (/T UW/). The consonant /T/ (unvoiced
% speech) looks like noise, while the vowel /UW/ (voiced speech) is
% characterized by a strong fundamental frequency.
%
[audioIn, fs] = audioread('Counting-16-44p1-mono-15secs.wav');
twoStart = 110e3;
twoStop = 135e3;
audioIn = audioIn(twoStart:twoStop);
timeVector = linspace((twoStart/fs),(twoStop/fs),numel(audioIn));
figure;
plot(timeVector,audioIn);
axis([(twoStart/fs) (twoStop/fs) -1 1]);
ylabel('Amplitude');
xlabel('Time (s)');
title('Utterance - Two')
sound(audioIn,fs);

%%
% The simplest method to distinguish between voiced and unvoiced speech is
% to analyze the zero crossing rate. A large number of zero crossings
% implies that there is no dominate low-frequency oscillation.
%
% Once you isolate a region of voiced speech, you can characterize it by
% estimating the pitch. This example uses 
% |<matlab:web(fullfile(docroot,'audio/ref/pitch.html'),'-helpbrowser') pitch>| 
% to estimate the pitch. It uses the default normalized 
% autocorrelation approach to calculating pitch.
%
% Apply pitch detection to the word "two" to see how pitch changes over
% time. This is known as the _pitch_ _contour_, and is characteristic to a
% speaker.
pD = audiopluginexample.SpeechPitchDetector;
[~,pitch] = process(pD,audioIn);

figure;
subplot(2,1,1);
plot(timeVector,audioIn);
axis([(110e3/fs) (135e3/fs) -1 1])
ylabel('Amplitude')
xlabel('Time (s)')
title('Utterance - Two')

subplot(2,1,2)
plot(timeVector,pitch,'*')
axis([(110e3/fs) (135e3/fs) 80 140])
ylabel('Pitch (Hz)')
xlabel('Time (s)');
title('Pitch Contour');

%%
% *Mel-Frequency Cepstrum Coefficients (MFCC)*
%
% Mel-Frequency Cepstrum Coefficients (MFCC) are popular features extracted
% from speech signals for use in recognition tasks. In the source-filter
% model of speech, MFCCs are understood to represent the filter (vocal
% tract). The frequency response of the vocal tract is relatively smooth,
% whereas the source of voiced speech can be modeled as an impulse train.
% The result is that the vocal tract can be estimated by the spectral
% envelope of a speech segment.
%
% The motivating idea of MFCC is to compress information about the vocal
% tract (smoothed spectrum) into a small number of coefficients based on an
% understanding of the cochlea.
%
% Although there is no hard standard for calculating MFCC, the basic steps
% are outlined by the diagram.
%
% <<SpeakerID02.png>>
% 
% The mel filterbank uses 10 linearly spaced triangular filters and then 
% logarithmically spaces them thereafter. The individual bands are weighted
% for even energy. Below is a visualization of a typical mel filterbank.
%
% <<SpeakerID03.png>>
%
% This example uses 
% |<matlab:web(fullfile(docroot,'audio/ref/mfcc.html'),'-helpbrowser') mfcc>| 
% to calculate the MFCCs for every file.
%
% A speech signal is dynamic in nature and changes over time. It is assumed   
% that speech signals are stationary on short time scales and 
% their processing is done in windows of 20-40 ms.
% This example uses a 30 ms window with a 75% overlap.

%% Data Set
% This example uses the Census Database (also known as AN4 Database) from
% the CMU Robust Speech Recognition Group [1]. The data set contains
% recordings of male and female subjects speaking words and numbers. The
% helper function in this section downloads it for you and converts the raw
% files to flac. The speech files are partitioned into subdirectories based
% on the labels corresponding to the speakers. If you are unable to
% download it, you can load a table of features from
% |HelperAN4TrainingFeatures.mat| and proceed directly to the *Training a
% Classifier* section. The features have been extracted from the same data
% set.
%
% Download and extract the speech files for 10 speakers (5 female and 5
% male) into a temporary directory using the |HelperAN4Download| function.
 
%dataDir = HelperAN4Download; % Path to data directory 
% change path as needed
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

%% plotting with pca to visualize our feature vectors 
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
    %[h, ~] = legend('show');
    %all = [all pcaTest];
    %legend('show');
    
    hold on;
end

l_list = {};
for i=1:size(names)
%lgd = legend(names);
    name= names{i};
    disp(names{i});
    l_list = [l_list, ['\color[rgb]{' num2str(CM(map(name), 1)) ',' num2str(CM(map(name), 2)) ',' num2str(CM(map(name), 3))  '} ' name]];
    
end 

lgd = legend(l_list);
lgd.FontSize = 14;
%colorbar
% [h, ~, plots] = legend(names);
% for idx = 1:length(h.String)
%     h.String{idx} = [CM(map(h.String{idx}),:) ' ' h.String{idx}]
% end

%legend([hbc{:}],'old','new');
hold off;

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
%
% Train the classifier and print the cross-validation accuracy.
% |<matlab:web(fullfile(docroot,'stats/classificationknn.crossval.html'),'-helpbrowser')
% crossval>| and
% |<matlab:web(fullfile(docroot,'stats/classificationpartitionedmodel.kfoldloss.html'),'-helpbrowser')
% kfoldloss>| are used to compute the cross-validation accuracy for the KNN
% classifier. Use
% |<matlab:web(fullfile(docroot,'stats/confusionmat.html'),'-helpbrowser')
% confusionmat>| to compute the confusion matrix (in percentage) and plot
% the result.
[trainedClassifier, validationAccuracy, confMatrix] = ...
    HelperTrainKNNClassifier(features);
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
heatmap(trainedClassifier.ClassNames, trainedClassifier.ClassNames, ...
    confMatrix);
title('Confusion Matrix');

%%
% You can also use the
% |<matlab:web(fullfile(docroot,'stats/classificationlearner-app.html'),'-helpbrowser')
% classificationLearner>| app to try out and compare various classifiers
% with your table of features.
%

%% Testing the Classifier
% In this section, you will test the trained KNN classifier with two speech
% signals from each of the ten speakers to see how well it behaves with
% signals that were not used to train it. 
%
% Read files, extract features from the test set, and normalize them.
lenDataTest = length(testDatastore.Files);
featuresTest = cell(lenDataTest,1);
for i = 1:lenDataTest
  [dataTest, infoTest] = read(testDatastore);
  len_ = size(dataTest);
  len_ = len_(1);
  dataTest = dataTest(1:(int16(len_)*1),:);
    % checking if the data gave more than one frame - cut it off
    len = size(dataTest);
    if len(2) > 1
        dataTest = dataTest(:,1);
        infoTest = infoTest(:,1);
    end
  featuresTest{i} = HelperComputePitchAndMFCC(dataTest,infoTest); 
  
end
featuresTest = vertcat(featuresTest{:});
featuresTest = rmmissing(featuresTest);
featuresTest{:,2:15} = (featuresTest{:,2:15}-m)./s; 
head(featuresTest)   % Display the first few rows

%%
% If you didn't download the AN4 database, you can load the table of
% features for test files from |HelperAN4TestingFeatures.mat|.
%
% The function |<matlab:edit('HelperTestKNNClassifier')
% HelperTestKNNClassifier>| performs the following actions for every file
% in |testDatastore|:
%
% # Read audio samples and compute pitch and MFCC features for each 30 ms
% frame, as described in the *Feature Extraction* section.
% # Predict the label (speaker) for each frame by calling |predict| on
% |trainedClassifier|.
% # For a given file, predictions are made for every frame. The most frequently 
% occuring label is declared as the predicted speaker for the file.  
% Prediction confidence is computed as the frequency of prediction of the label 
% divided by the total number of voiced frames in the file.
result = HelperTestKNNClassifier(trainedClassifier, featuresTest)

%%
% The predicted speakers match the expected speakers for all files under
% test.
%
% The experiment was repeated using an internally developed dataset. The
% dataset consists of 20 speakers with each speaker speaking multiple
% sentences from the Harvard sentence list [2]. For 20 speakers, the
% validation accuracy was found to be 89%.

%% References
%
% [1] http://www.speech.cs.cmu.edu/databases/an4/
%
% [2] http://en.wikipedia.org/wiki/Harvard_sentences

displayEndOfDemoMessage(mfilename)
