function [feature,distance,threshold] = compute_Hellinger_distance(features, labels, numBins)
%Function: compute_Hellinger_distance
%Form: [feature,distance,threshold] = compute_Hellinger_distance(features, labels, numBins)
%Description: Compute Hellinger distances at discrete intervals for given
%       features
%Parameters:
%   features: I X F numeric matrix where I is the number of instances and F
%       is the number of features. Each row represents one training instance
%       and each column represents the value of one of its corresponding features
%   labels: I x 1 numeric matrix where I is the number of instances. Each
%       row is the label of a specific training instance and corresponds to
%       the same row in features
%   numBins: Number of bins for discretizing numeric features.
%Output:
%   feature: Index of feature with maximum distance
%   distance: Maximum Hellinger distance of all given features at the
%       optimal threshold
%   threshold: Optimal threshold for feature which results in the maximum Hellinger distance

minVals = min(features);
maxVals = max(features);
binSize = (maxVals - minVals) ./ numBins;
Tplus = sum(labels == 1);
Tminus = sum(labels == 0);
minVals = repmat(minVals,[numBins - 1,1]);
binSizes = repmat(binSize,[numBins - 1,1]) .* repmat([1:numBins-1],[length(binSize),1])';
thresholds = minVals + binSizes;
labels = repmat(labels,[1,size(thresholds)]);
[r,c] = size(features);
features = reshape(features,[r 1 c]);
features = repmat(features,[1,size(thresholds,1),1]);
[r,c] = size(thresholds);
thresholds = reshape(thresholds,[1 r c]);
thresholds = repmat(thresholds,[size(labels,1),1]);
Tlplus = sum(labels & (features <= thresholds));
Tlminus = sum(~labels & (features <= thresholds));
Trplus = sum(labels & (features > thresholds));
Trminus = sum(~labels & (features > thresholds));
distance = (sqrt(Tlplus ./ Tplus) - sqrt(Tlminus ./ Tminus)).^2 + (sqrt(Trplus ./ Tplus) - sqrt(Trminus ./ Tminus)).^2;
[~,r,c] = size(distance);
distance = reshape(distance,[r,c]);
[distance,index] = max(distance);
[distance,feature] = max(distance);
threshold = thresholds(1,index(feature),feature);

end