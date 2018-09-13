function model = HDDT(features,labels,model,numBins,cutoff,memThresh,memSplit)
%Function: HDDT
%Form: model = HDDT(features,labels,model,numBins,cutoff,memThresh,memSplit)
%Description: Recursively build Hellinger Distance Decision Tree
%Parameters:
%   features: I X F numeric matrix where I is the number of instances and F
%       is the number of features. Each row represents one training instance
%       and each column represents the value of one of its corresponding features
%   labels: I x 1 numeric matrix where I is the number of instances. Each
%       row is the label of a specific training instance and corresponds to
%       the same row in features
%   numBins: Number of bins for discretizing numeric features.
%   cutoff: Number representing maximum number of instances in a
%       leaf node.
%   memSplit: If features matrix is large, compute discretization splits
%       iteratively in batches of size memSplit instead all at once.
%   memThresh: If features matrix is large, compute discretization splits
%       iteratively in batches of size memSplit only if number of instances
%       in branch is greater than memThresh.
%Output:
%   model: a Hellinger Distance Decision Subtree model

numSamples = size(features,1);
if length(unique(labels)) == 1 || numSamples <= cutoff
    model.complete = true;
    model.label = mode(labels);
    return
end
numFeatures = size(features,2);

selectedFeature = -1;
selectedThreshold = -1;
selectedDistance = -1;

if numSamples <= memThresh
    memSplit = 1;
end

for i = 1:floor(numFeatures ./ memSplit):numFeatures
    maxIndex = min(numFeatures,i+floor(numFeatures ./ memSplit)-1);
    featureIndices = [i:maxIndex];
    featuresTemp = features(:,i:maxIndex);
    [featureIndex,featureDistance,featureThreshold] = compute_Hellinger_distance(featuresTemp, labels, numBins);
    if featureDistance > selectedDistance
        selectedFeature = featureIndices(featureIndex);
        selectedThreshold = featureThreshold;
        selectedDistance = featureDistance;
    end
end

model.threshold = selectedThreshold;
model.feature = selectedFeature;
featuresLeft = features(features(:,selectedFeature) <= selectedThreshold,:);
labelsLeft = labels(features(:,selectedFeature) <= selectedThreshold,:);
featuresRight = features(features(:,selectedFeature) > selectedThreshold,:);
labelsRight = labels(features(:,selectedFeature) > selectedThreshold,:);
if size(featuresLeft,1) == numSamples || size(featuresRight,1) == numSamples
    model.complete = true;
    model.label = mode(labels);
    return
end
clear features;
clear labels;
modelLeft = HellingerTreeNode;
modelRight = HellingerTreeNode;
model.leftBranch = HDDT(featuresLeft,labelsLeft,modelLeft,numBins,cutoff,memSplit,memThresh);
clear featuresLeft;
clear labelsLeft;
model.rightBranch = HDDT(featuresRight,labelsRight,modelRight,numBins,cutoff,memSplit,memThresh);
clear featuresRight;
clear labelsRight;
model.complete = false;

end