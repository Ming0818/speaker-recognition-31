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
hold on;

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
finish=false;
set(gcf,'CurrentCharacter','@'); % set to a dummy character
while ~finish
  audioCluster(recObj);
  % check for keys
  k=get(gcf,'CurrentCharacter');
  if k~='@' % has it changed from the dummy character?
    set(gcf,'CurrentCharacter','@'); % reset the character
    % now process the key as required
    if k=='q', finish=true; end
  end
end


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




