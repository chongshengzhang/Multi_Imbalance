% Reference:	
% Name: fitHellingerTreeMC.m
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

function model = fitHellingerTreeMC(features, labels, numBins, cutoff, memSplit, memThresh)

% Parameters:
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
% Output:
%   model: a trained MCHDDT model

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
model = HDDTMC(features,labels,model,numBins,cutoff,memThresh,memSplit);

end

