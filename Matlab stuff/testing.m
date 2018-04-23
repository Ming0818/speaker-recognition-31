
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




