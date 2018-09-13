function model = fit_Hellinger_tree(features, labels, numBins, cutoff, memSplit, memThresh)
%Function: fit_Hellinger_tree
%Form: model = fit_Hellinger_tree(features, labels, numBins, cutoff, memSplit, memThresh)
%Description: Train a single Hellinger Distance Decision Tree
%Parameters:
%   features: I X F numeric matrix where I is the number of instances and F
%       is the number of features. Each row represents one training instance
%       and each column represents the value of one of its corresponding features
%   labels: I x 1 numeric matrix where I is the number of instances. Each
%       row is the label of a specific training instance and corresponds to
%       the same row in features
%   numBins (optional): Number of bins for discretizing numeric features. 
%        Default: 100
%   cutoff (optional): Number representing maximum number of instances in a
%       leaf node. Default: 10 if more than ten instances, 1 otherwise
%   memSplit (optional): If features matrix is large, compute discretization splits
%       iteratively in batches of size memSplit instead all at once. Default: 1
%   memThresh (optional): If features matrix is large, compute discretization splits
%       iteratively in batches of size memSplit only if number of instances
%       in branch is greater than memThresh. Default: 1
%Output:
%   model: a trained Hellinger Distance Decision Tree model

[numInstances,numFeatures] = size(features);

if numInstances <= 1
    msgID = 'fit_Hellinger_tree:notEnoughData';
    msg = 'Feature array is empty or only instance exists';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if numFeatures == 0
    msgID = 'fit_Hellinger_tree:noData';
    msg = 'No feature data';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if size(labels,1) ~= numInstances
    msgID = 'fit_Hellinger_tree:mismatchInstanceSize';
    msg = 'Number of instances in feature matrix and label matrix do not match';
    causeException = MException(msgID,msg);
    throw(causeException);
end

labelIDs = unique(labels);
if length(labelIDs) ~= 2 || ismember(0,ismember(labelIDs,[0 1]))
    msgID = 'fit_Hellinger_tree:improperLabels';
    msg = 'Labels must be either 0 or 1; Label array may only contain a single label value';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if nargin < 3
    numBins = 100;
end

if nargin < 4
    cutoff = 10;
    if numInstances <= 10
        cutoff = 1;
    end
end

if nargin < 5
    memSplit = 1;
end

if nargin < 6
    memThresh = 1;
end

if numBins <= 0
    msgID = 'fit_Hellinger_tree:numBinsNonpositive';
    msg = 'numBins must be positive';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if cutoff <= 0
    msgID = 'fit_Hellinger_tree:cutoffNonpositive';
    msg = 'cutoff must be positive';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if memSplit <= 0
    msgID = 'fit_Hellinger_tree:memSplitNonpositive';
    msg = 'memSplit must be positive';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if memThresh <= 0
    msgID = 'fit_Hellinger_tree:memThreshNonpositive';
    msg = 'memThresh must be positive';
    causeException = MException(msgID,msg);
    throw(causeException);
end

model = HellingerTreeNode;
model = HDDT(features,labels,model,numBins,cutoff,memThresh,memSplit);

end

