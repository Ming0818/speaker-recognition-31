function [dissimilarity] = calc_dissimilarity(test,profile)
%calc_dissimilarity takes a test matrix and profile matrix and calculates
%the average distance from each test vector to their corresponding vector
%in the profile matrix. The corresponding vector is the nearest unmatched
%vector.

%test is 2d matrix: codebook_size X num_features

codebook_size = size(text,1);
num_features = size(test,2);

size(test)
size(profile)
[dists, ids] = pdist2(test,profile,'euclidean','Smallest',1);
dissimilarity = nanmean(dists);

end



