%% testing real-time audio recording
% this assumes you've already run 'running.m', so you have the proper
% variables saved in your workspace

% plotting the MFCC vectors by cluster
% figure();
% plot3(pcaData(ids==1,1),pcaData(ids==1,2),pcaData(ids==1,3),'o');
% hold on;
% for i = 2:max(ids)
%     plot3(pcaData(ids==i,1),pcaData(ids==i,2),pcaData(ids==i,3),'o');
%     disp(i)
% end
% title('Scatter of MFCC Vectors Colored by Clustering')
% 
% legend
% hold on;

% plotting the MFCC vectors by speaker
% figure();
% c = pcaDataCell{2,1};
% plot3(c(:,1),c(:,2),c(:,3),'o');
% hold on;
% for i = 2:size(pcaDataCell,2) %plot data points
%     c = pcaDataCell{2,i};
%     plot3(c(:,1),c(:,2),c(:,3),'o');
% end
% 
%  for i = 1:size(pcaDataCell,2) %Plot averages for each speaker
%     c = pcaDataCell{3,i};
%     plot3(c(:,1),c(:,2),c(:,3),'o');
% end
% title('Scatter of MFCC Vectors Colored By Speaker')
% hold on;

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

lgd = legend(l_list,'AutoUpdate','off');
lgd.FontSize = 14;
%colorbar
% [h, ~, plots] = legend(names);
% for idx = 1:length(h.String)
%     h.String{idx} = [CM(map(h.String{idx}),:) ' ' h.String{idx}]
% end

%legend([hbc{:}],'old','new');
%hold off;

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

%%

files = ["jake.wav" "tammany.wav" "lindsay.wav" "jake-hobbit.wav", ...
    "tammany-hobbit.wav", "lindsay-hobbit.wav", "scharf-hobbit.m4a", ...
    "michael-hobbit.m4a"];
numfiles = size(files,2);


%% Clean

figure(3);
x = audioread('jake.wav');
x = x(:,1);
x = x(1:10000,1);
[W,s,v]=svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');
plot(t,v(:,1));
hold on;
maxAmp = max(v(:,1));
plot(t,v(:,2),'r'); xlabel('time'); ylabel('amplitude'); axis([0 0.005 -maxAmp maxAmp]); legend('isolated tone 1', 'isolated tone 2');
hold off;


%%


%Calculate MFCCs
mfccs = cell(1,numfiles);
for i = 1:numfiles
    filename = files(1,i);
    
    mffcs_for_filename; %run on file 'filename'
        
    mfccs(1,i) = {MFCCs}; %get MFCCs
    
 
end
close all;


%% Run Clustering and analyze

clustersToTest = 10;

for k = 1:10
    % Put all data together in one vector
    combined = [];
    for i = 1:size(mfccs,2)
        combined = [combined mfccs{1,i} ];
    end

    [ids, centroids, sums] = kmeans(combined', k); % run kmeans
    sse = sum(sums.^ 2);
    
    sses(k) = sse;
end

subplot(2,1,1)
plot(sses);
subplot(2,1,2)
semilogy(sses);


%% 




