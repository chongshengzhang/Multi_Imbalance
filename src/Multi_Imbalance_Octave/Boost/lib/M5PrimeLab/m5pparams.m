function trainParams = m5pparams(modelTree, minLeafSize, minParentSize, ...
    prune, smoothingK, splitThreshold, aggressivePruning, maxDepth, ...
    eliminateTerms, vanillaSDR, extractRules)
% m5pparams
% Creates configuration for building M5' trees or decision rules. The
% output structure is for further use with m5pbuild and m5pcv functions.
%
% Call:
%   trainParams = m5pparams(modelTree, minLeafSize, minParentSize, ...
%       prune, smoothingK, splitThreshold, aggressivePruning, maxDepth, ...
%       eliminateTerms, vanillaSDR, extractRules)
%
% All the input arguments of this function are optional. Empty values are
% also accepted (the corresponding defaults will be used).
% It is quite possible that the default values for minLeafSize and
% minParentSize will be far from optimal for your data.
% For a typical configuration of ensembles of regression trees (whether
% Bagging, Random Forests, or Extra-Trees), call
% trainParams = m5pparams(false, 1, 5, false, 0, 1E-6);
%
% Input:
%   modelTree     : Whether to build a model tree (true) or a regression
%                   tree (false) (default value = true). Model trees
%                   combine a conventional regression tree with the
%                   possibility of linear regression functions at the
%                   leaves. However, note that whether a leaf node actually
%                   contains more than just a constant, depends on pruning
%                   and smoothing (if both are disabled, a model tree will
%                   not differ from a regression tree).
%   minLeafSize   : The minimum number of training observations a leaf node
%                   may represent. If prune = true, allowed minimum is 2.
%                   Otherwise, allowed minimum is 1. Default value = 2
%                   (Wang & Witten, 1997). If built trees contain too many
%                   too small leaves (especially in the top layers of the
%                   tree), consider increasing this number. This will also
%                   result in smaller trees that are less sensitive to
%                   noise (but can also be underfitted).
%   minParentSize : The minimum number of observations a node must have to
%                   be considered for splitting, i.e., the minimum number
%                   of training observations an interior node may
%                   represent. Default value = minLeafSize * 2 (Wang &
%                   Witten, 1997). Values lower than that are not allowed.
%                   If built trees are too large or overfit the data,
%                   consider increasing this number - this will result in
%                   smaller trees that are less sensitive to noise (but can
%                   also be underfitted). For ensembles of unpruned trees,
%                   the typical value is 5 (with minLeafSize = 1) (Breiman,
%                   2001).
%   prune         : Whether to prune the tree (default value = true).
%                   Pruning is done by eliminating leaves and subtrees in
%                   regression trees and model trees as well as, if
%                   eliminateTerms = true, by eliminating terms in models
%                   of model trees (using sequential backward selection
%                   algorithm), if doing so improves the estimated error.
%   smoothingK    : Smoothing parameter. Set to 0 to disable smoothing
%                   (default). Smoothing is usually not recommended for
%                   regression trees but can be useful for model trees. It
%                   tries to compensate for sharp discontinuities occurring
%                   between adjacent nodes of the tree. The larger the
%                   value compared to the number of observations reaching
%                   the nodes, the more pronounced is the smoothing. In
%                   case studies by Quinlan, 1992, as well as Wang &
%                   Witten, 1997, this almost always had a positive effect
%                   on model trees (with smoothingK = 15). Smoothing is
%                   performed after tree building and pruning, therefore
%                   this parameter does not influence those processes.
%                   Unfortunately, smoothed trees are harder to interpret.
%   splitThreshold : A node is not split if the standard deviation of the
%                   values of response variable at the node is less than
%                   splitThreshold of the standard deviation of response
%                   variable for the entire training data (default value
%                   = 0.05 (i.e., 5%) (Wang & Witten, 1997)).
%   aggressivePruning : By default, pruning is done as proposed by Quinlan,
%                   1992, and Wang & Witten, 1997, but you can also employ
%                   more aggressive pruning, similar to the one that is
%                   implemented in Weka's version of M5' (Hall et al.,
%                   2009). Simply put, in the aggressive pruning version,
%                   while estimating error of a subtree, one penalizes not
%                   only the number of parameters of regression models at
%                   its leaves but also its total number of splits.
%                   Aggressive pruning produces smaller trees that are less
%                   sensitive to noise and potentially easier to interpret.
%                   However, this can also result in underfitting. (default
%                   value = false)
%   maxDepth      : Maximum depth of a tree. Controlling tree complexity
%                   using this parameter is not typical for M5' trees.
%                   (default value = Inf, i.e., no limitation)
%   eliminateTerms : Whether to eliminate terms of models in model trees
%                   using the sequential backward selection algorithm
%                   (default value = true). The parameter has no effect if
%                   prune = false.
%   vanillaSDR    : Whether to calculate the Standard Deviation Reduction
%                   criterion using the "vanilla" formula from Quinlan,
%                   1992 or the updated formula from Wang & Witten, 1997
%                   (default value = false, i.e., the updated formula is
%                   used). The difference between those two formulas is
%                   that the updated one tries to take into account the
%                   number of missing values for each input variable, as
%                   well as the number of categories for each categorical
%                   input variable. See equations (1) and (5) in Wang &
%                   Witten, 1997. Note that the updated formula is actually
%                   fairly ad hoc and not well studied.
%                   This parameter has no effect if the data has no missing
%                   values and no categorical variables with more than two
%                   categories.
%   extractRules  : M5' trees can also be used for generating decision
%                   rules. M5PrimeLab provides two methods for doing it.
%                   Set extractRules = 1 to extract rules from one tree
%                   directly. Each leaf is made into a rule by making a
%                   conjunction of all the tests encountered on the path
%                   from the root to that leaf. This produces rules that
%                   are unambiguous in that it doesn’t matter in what order
%                   they are executed. The rule set always makes exactly
%                   the same predictions as the original tree, even with
%                   unknown values and smoothing.
%                   Set extractRules = 2 to use the M5'Rules method (Holmes
%                   et al., 1999). With this method, the rules are
%                   generated iteratively. In each iteration, a new tree is
%                   built using the training data and one leaf that has the
%                   largest data coverage is made into a rule. Then the
%                   tree is discarded and all observations covered by the
%                   rule are removed from the data. The process is repeated
%                   until the data is empty. M5'Rules produces smaller rule
%                   sets than the simple extraction method, however it
%                   cannot use the M5' smoothing technique (parameter
%                   smoothingK is ignored).
%                   (default value = 0, i.e., no rules are extracted)
%
% Output:
%   trainParams   : A structure of parameters for further use with m5pbuild
%                   and m5pcv functions containing the provided values (or
%                   defaults, if not provided).

% =========================================================================
% M5PrimeLab: M5' regression tree, model tree, and tree ensemble toolbox for Matlab/Octave
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2010-2016  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

% Last update: August 5, 2016

if (nargin < 1) || isempty(modelTree)
    trainParams.modelTree = true;
else
    if ischar(modelTree)
        error('Bad argument value.');
    end
    trainParams.modelTree = modelTree;
end

if (nargin < 2) || isempty(minLeafSize)
    trainParams.minLeafSize = 2;
else
    trainParams.minLeafSize = max(1, minLeafSize);
end

if (nargin < 3) || isempty(minParentSize)
    trainParams.minParentSize = trainParams.minLeafSize * 2;
else
    trainParams.minParentSize = max(trainParams.minLeafSize * 2, minParentSize);
end

if (nargin < 4) || isempty(prune)
    trainParams.prune = true;
else
    trainParams.prune = prune;
end

if (nargin < 5) || isempty(smoothingK)
    trainParams.smoothingK = 0;
else
    trainParams.smoothingK = max(0, smoothingK);
end

if (nargin < 6) || isempty(splitThreshold)
    trainParams.splitThreshold = 0.05;
else
    trainParams.splitThreshold = max(0, splitThreshold);
end

if (nargin < 7) || isempty(aggressivePruning)
    trainParams.aggressivePruning = false;
else
    trainParams.aggressivePruning = aggressivePruning;
end

if (nargin < 8) || isempty(maxDepth)
    trainParams.maxDepth = Inf;
else
    trainParams.maxDepth = max(0, floor(maxDepth));
end

if (nargin < 9) || isempty(eliminateTerms)
    trainParams.eliminateTerms = true;
else
    trainParams.eliminateTerms = eliminateTerms;
end

if (nargin < 10) || isempty(vanillaSDR)
    trainParams.vanillaSDR = false;
else
    trainParams.vanillaSDR = vanillaSDR;
end

if (nargin < 11) || isempty(extractRules)
    trainParams.extractRules = 0;
else
    trainParams.extractRules = max(0, min(2, floor(extractRules)));
end

return
