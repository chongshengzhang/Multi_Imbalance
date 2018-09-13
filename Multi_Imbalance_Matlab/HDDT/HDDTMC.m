% Reference：
% Hoens, T. R., Qian, Q., Chawla, N. V., et al. (2012). Building decision trees for the multi-class imbalance
% problem. Advances in Knowledge Discovery and Data Mining. Springer Berlin Heidelberg, 2012 (PP. 122-134).
%
% (1) About HDDT:
% The Hellinger distance decision trees (HDDT) method is a classification algorithm based on decision trees
% for binary imbalanced data. When building a decision tree, the splitting criterion used in the HDDT is the
% Hellinger distance, see the formula (1) in our KBS paper, where |X_+| and |X_-|) respectively represents
% the total number of samples with positive (negative) labels, whereas |X_(+j)| and X_(-j)| respectively
% denotes the number of positive (negative) examples with the j-th value of the current feature.
%
% For each node to be splitted, HDDT calculates the Hellinger distance for each attribute on the node, then
% splits the node using the feature with the maximum Hellinger distance.
% See the compute_Hellinger_distance() function, line 76.
%
% (2)About Multi-class HDDT.
% The Multi-Class HDDT method, successively takes one or a pair of classes as the positive class and the
% rest as negative class, when calculating the Hellinger distance for each feature. It next selects the
% maximum Hellinger value for this feature. Finally, it obtains the maximum Hellinger value for each feature,
% then the feature with maximum Hellinger distance will be used to split the node, see lines 71-81.
% Then in lines 99-106, after determining the best split feature, it recursively build the child trees.

function model = HDDTMC(features,labels,model,numBins,cutoff,memThresh,memSplit)

% Description: Recursively build Hellinger Distance Decision Tree
% Parameters:
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
% Output:
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





label=unique(labels);%


for cl = 1:length(label) %%for each pair of subsets of C: C1 ? C, C2 = C \ C1
    labelsCL = labels == label(cl);%
    
    for i = 1:floor(numFeatures ./ memSplit):numFeatures
        maxIndex = min(numFeatures,i+floor(numFeatures ./ memSplit)-1);
        featureIndices = [i:maxIndex];
        featuresTemp = features(:,i:maxIndex);
        
        [featureIndex,featureDistance,featureThreshold] = compute_Hellinger_distance(featuresTemp, labelsCL, numBins);%
        if featureDistance > selectedDistance
            selectedFeature = featureIndices(featureIndex);
            selectedThreshold = featureThreshold;
            selectedDistance = featureDistance;
        end
    end
    
end%

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
model.leftBranch = HDDTMC(featuresLeft,labelsLeft,modelLeft,numBins,cutoff,memSplit,memThresh);
clear featuresLeft;
clear labelsLeft;
model.rightBranch = HDDTMC(featuresRight,labelsRight,modelRight,numBins,cutoff,memSplit,memThresh);
clear featuresRight;
clear labelsRight;
model.complete = false;

end
