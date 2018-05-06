% this is the old speaker classification method (not including the
% subclusters - it's un-normalized, so to normalize uncomment out line 10)
%% classification based on given mfccs for one speaker
% means must be a containers.Map object of the form label: mean
function label = spkr_classify(mfccs, map, plot)

%map = containers.Map(names, indices);

% normalizing the given speaker MFCCs
%mfccs = (mfccs-nanmean(mfccs))./std(mfccs);
pcaCoeffs = pca(mfccs');

% taking the mean
mfccs = nanmean(mfccs,2);

% if we want to plot the mean - plot the pca
if plot
    % running pca on the mfccs for plotting purposes
    pcaTest = mfccs'*pcaCoeffs;
    plot3(pcaTest(:,1), pcaTest(:, 2), pcaTest(:, 3), '*', 'color', 'k');
end

% compare to given audio profiles - return the label of whichever is
% closest
len = size(map);
labels = keys(map);
distances = zeros(1,len(1));
for i=1:len(1)
    % compute euclidean distance to nearest mean
    dist = norm(mfccs-map(labels{i}));
    distances(i) = dist;
end

[min_val, min_ind] = min(distances);
disp(distances);
label = labels{min_ind};

end
