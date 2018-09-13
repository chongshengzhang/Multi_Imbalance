function trainParamsEnsemble = m5pparamsensemble(numTrees, numVarsTry, ...
    withReplacement, inBagFraction, extraTrees, getOOBError, getVarImportance, ...
    getOOBContrib, verboseNumIter)
% m5pparamsensemble
% Creates configuration for building ensembles of M5' trees using Bagging,
% Random Forests, or Extra-Trees. The output structure is for further use
% with m5pbuild and m5pcv functions.
%
% Call:
%   trainParamsEnsemble = m5pparamsensemble(numTrees, numVarsTry, ...
%       withReplacement, inBagFraction, extraTrees, getOOBError, ...
%       getVarImportance, getOOBContrib, verboseNumIter)
%
% All the input arguments of this function are optional. Empty values are
% also accepted (the corresponding defaults will be used). The first
% five arguments control the behaviour of the ensemble building method. The
% last four arguments enable getting additional information.
% The default values are prepared for building Random Forests. Changes
% required for a Bagging configuration: numVarsTry = 0. Changes required
% for a typical Extra-Trees configuration: numVarsTry = 0, extraTrees =
% true.
% Remember to configure how individual trees are built for the ensemble
% (see description of m5pparams). See user's manual for examples of usage.
%
% Input:
%   numTrees      : Number of trees to build (default value = 100). Should
%                   be set so that every data observation gets predicted at
%                   least a few times.
%   numVarsTry    : Number of input variables randomly sampled as
%                   candidates at each split in a tree. Set to -1 (default)
%                   to automatically sample one third of the variables
%                   (typical for Random Forests in regression). Set to 0 to
%                   use all variables (typical for Bagging and Extra-Trees
%                   in regression). Set to a positive integer if you want
%                   some other number of variables to sample. To select a
%                   good value for numVarsTry in Random Forests, Leo
%                   Breiman suggests trying the default value and trying a
%                   value twice as high and half as low (Breiman, 2002).
%                   In Extra-Trees, this parameter is also called attribute
%                   selection strength (Geurts et al., 2006).
%                   Note that while using this parameter, function
%                   m5pbuild takes the total number of input variables
%                   directly from supplied training data set, before any
%                   synthetic binary variables are made (from categorical
%                   variables with more than two categories). Also note
%                   that m5pbuild will always round the numVarsTry value
%                   down.
%   withReplacement : Should sampling of in-bag observations for each tree
%                   be done with (true) or without (false) replacement?
%                   Both, Bagging and Random Forests typically use sampling
%                   with replacement. (default value = true)
%   inBagFraction : The fraction of the total number of observations to be
%                   sampled for in-bag set. Default value = 1, i.e., the
%                   in-bag set will be the same size as the original data
%                   set. This is the typical setting for both, Bagging and
%                   Random Forests. Note that for sampling without
%                   replacement inBagFraction should be lower than 1 so
%                   that out-of-bag set is not empty.
%   extraTrees    : Set to true to build Extra-Trees (default = false). If
%                   enabled, parameters withReplacement, inBagFraction,
%                   getOOBError, and getVarImportance are ignored. This is
%                   because Extra-Trees method does not use out-of-bag
%                   data, i.e., all trees are build using the whole
%                   available training data set.
%   getOOBError   : Whether to perform out-of-bag error calculation to
%                   estimate prediction error of the ensemble (default
%                   value = true). The result will be stored in the output
%                   argument ensembleResults of function m5pbuild. Disable
%                   for speed.
%   getVarImportance : Whether to assess importance of input variables (by
%                   calculating the average increase in error when out-of-
%                   bag data of a variable is permuted) and how many times
%                   the data is permuted per tree for the assessment.
%                   Default value = 1. Set to 0 to disable and gain some
%                   speed. Numbers larger than 1 can give slightly more
%                   stable estimate, but the process is even slower. The
%                   result will be stored in the output argument
%                   ensembleResults of function m5pbuild.
%   getOOBContrib : Whether to compute input variable contributions in
%                   out-of-bag data according to the Forest Floor
%                   methodology. Available only for ensembles of unsmoothed
%                   regression trees and only if m5pbuild is called with
%                   keepNodeInfo = true. The result will be stored in the
%                   output argument ensembleResults of function m5pbuild.
%                   For details, see description of OOBContrib.
%   verboseNumIter : Set to some positive integer to print progress every
%                   verboseNumIter trees. Set to 0 to disable. (default
%                   value = 50)
%
% Output:
%   trainParamsEnsemble : A structure of parameters for further use with
%                   m5pbuild and m5pcv functions containing the provided
%                   values (or defaults, if not provided).
%
% Remarks:
% See the note in Section 1 of the user's manual on the most important
% difference between the implementation of Extra-Trees in M5PrimeLab and
% standard Extra-Trees.

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

if (nargin < 1) || isempty(numTrees)
    trainParamsEnsemble.numTrees = 100;
else
    if ischar(numTrees)
        error('Bad argument value.');
    end
    trainParamsEnsemble.numTrees = max(1, floor(numTrees));
end

if (nargin < 2) || isempty(numVarsTry)
    trainParamsEnsemble.numVarsTry = -1;
else
    trainParamsEnsemble.numVarsTry = numVarsTry;
end

if (nargin < 3) || isempty(withReplacement)
    trainParamsEnsemble.withReplacement = true;
else
    trainParamsEnsemble.withReplacement = withReplacement;
end

if (nargin < 4) || isempty(inBagFraction)
    trainParamsEnsemble.inBagFraction = 1;
else
    trainParamsEnsemble.inBagFraction = max(0, min(1, inBagFraction));
end

if (nargin < 5) || isempty(extraTrees)
    trainParamsEnsemble.extraTrees = false;
else
    trainParamsEnsemble.extraTrees = extraTrees;
end

if (nargin < 6) || isempty(getOOBError)
    trainParamsEnsemble.getOOBError = true;
else
    trainParamsEnsemble.getOOBError = getOOBError;
end

if (nargin < 7) || isempty(getVarImportance)
    trainParamsEnsemble.getVarImportance = 1;
else
    trainParamsEnsemble.getVarImportance = max(0, getVarImportance);
end

if (nargin < 8) || isempty(getOOBContrib)
    trainParamsEnsemble.getOOBContrib = true;
else
    trainParamsEnsemble.getOOBContrib = getOOBContrib;
end

if (nargin < 9) || isempty(verboseNumIter)
    trainParamsEnsemble.verboseNumIter = 50;
else
    trainParamsEnsemble.verboseNumIter = max(0, floor(verboseNumIter));
end

return
