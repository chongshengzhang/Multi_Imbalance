% Reference:	
% Name: HDDT.m
% 
% Purpose: Recursively build Hellinger Distance Decision Tree
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
    [featureIndex,featureDistance,featureThreshold] = computeHellingerDistance(featuresTemp, labels, numBins);
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
modelLeft = hellingerTreeNode;
modelRight = hellingerTreeNode;
model.leftBranch = HDDT(featuresLeft,labelsLeft,modelLeft,numBins,cutoff,memSplit,memThresh);
clear featuresLeft;
clear labelsLeft;
model.rightBranch = HDDT(featuresRight,labelsRight,modelRight,numBins,cutoff,memSplit,memThresh);
clear featuresRight;
clear labelsRight;
model.complete = false;

end