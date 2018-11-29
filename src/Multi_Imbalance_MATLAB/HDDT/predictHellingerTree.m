% Reference:	
% Name: predictedHellingerTree.m
% 
% Purpose: Predict labels using trained Hellinger Distance Decision Tree
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

function predicted_classes = predictHellingerTree(model,features)
%Function: predict_Hellinger_tree
%Form: predicted_classes = predict_Hellinger_tree(model,features)
%Description: 
%Parameters:
%   model: a trained Hellinger Distance Decision Tree model
%   features: I X F numeric matrix where I is the number of instances and F
%       is the number of features. Each row represents one training instance
%       and each column represents the value of one of its corresponding features
%Output:
%   predicted_classes: I X 1 matrix where each row represents a predicted label of the corresponding feature set 

[numInstances,numFeatures] = size(features);

if numInstances <= 0
    msgID = 'predict_Hellinger_tree:notEnoughData';
    msg = 'Feature array is empty or only instance exists';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if numFeatures == 0
    msgID = 'predict_Hellinger_tree:noData';
    msg = 'No feature data';
    causeException = MException(msgID,msg);
    throw(causeException);
end

initialModel = model;
predicted_classes = zeros(size(features,1),1);
for i = 1:1:size(features,1)
    model = initialModel;
    complete = model.complete;
    while ~complete
        if features(i,model.feature) <= model.threshold
            model = model.leftBranch;
        else
            model = model.rightBranch;
        end
        complete = model.complete;
    end
    predicted_classes(i) = model.label;
end
end