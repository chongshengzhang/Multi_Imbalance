% Reference:	
% Name: fitHellingerTree.m
% 
% Purpose: Train a single Hellinger Distance Decision Tree
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

model = hellingerTreeNode;
model = HDDT(features,labels,model,numBins,cutoff,memThresh,memSplit);

end

