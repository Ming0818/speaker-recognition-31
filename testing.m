
files = ["jake.wav" "tammany.wav" "lindsay.wav"];
numfiles = size(files,2);


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




