%Class: HellingerTreeNode
%Description: Representation for single tree node in Hellinger Distance
%       Decision Tree
%Properties:
%   threshold: Value of threshold for branching
%   feature: Index of feature for branching
%   leftBranch: Node for value <= threshold
%   rightBranch: Node for value > threshold
%   complete: Boolean for whether node is a leaf
%   label: Label to assign if node reached

classdef HellingerTreeNode
    properties
        threshold
        feature
        leftBranch
        rightBranch
        complete
        label
    end
end