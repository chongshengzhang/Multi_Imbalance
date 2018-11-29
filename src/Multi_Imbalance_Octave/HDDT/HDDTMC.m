% Reference:	
% Name: HDDTMC.m
% 
% Purpose: determining the best split feature, it recursively build the child trees.
% 
% Authors: Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% Copyright: (c) 2018 Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% This file is a part of Multi_Imbalance software, a software package for multi-class Imbalance learning. 
% 
% Multi_Imbalance software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
% as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
%
% Multi_Imbalance software is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program. 
% If not, see <http://www.gnu.org/licenses/>.
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
        
        [featureIndex,featureDistance,featureThreshold] = computeHellingerDistance(featuresTemp, labelsCL, numBins);%
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
modelLeft = hellingerTreeNode;
modelRight = hellingerTreeNode;
model.leftBranch = HDDTMC(featuresLeft,labelsLeft,modelLeft,numBins,cutoff,memSplit,memThresh);
clear featuresLeft;
clear labelsLeft;
model.rightBranch = HDDTMC(featuresRight,labelsRight,modelRight,numBins,cutoff,memSplit,memThresh);
clear featuresRight;
clear labelsRight;
model.complete = false;

end
