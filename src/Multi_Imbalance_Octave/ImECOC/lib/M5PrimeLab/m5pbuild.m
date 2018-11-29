function [model, time, ensembleResults] = m5pbuild(Xtr, Ytr, trainParams, ...
    isBinCat, trainParamsEnsemble, keepNodeInfo, verbose)
% m5pbuild
% Builds M5' regression tree, model tree, or ensemble of trees. Trees can
% also be linearized into decision rules. For tree ensembles, can also
% assess input variable importances as well as provide data for ensemble
% interpretation.
%
% Call:
%   [model, time, ensembleResults] = m5pbuild(Xtr, Ytr, trainParams, ...
%       isBinCat, trainParamsEnsemble, keepNodeInfo, verbose)
%
% All the input arguments, except the first two, are optional. Empty values
% are also accepted (the corresponding defaults will be used).
%
% Input:
%   Xtr, Ytr      : Xtr is a matrix with rows corresponding to
%                   observations and columns corresponding to input
%                   variables. Ytr is a column vector of response values.
%                   Input variables can be continuous, binary, as well as
%                   categorical (indicated using isBinCat). All values must
%                   be numeric. Categorical variables with more than two
%                   categories will be automatically replaced with
%                   synthetic binary variables (in accordance with the M5'
%                   method). Missing values in Xtr must be indicated as
%                   NaN.
%   trainParams   : A structure of training parameters for the algorithm.
%                   If not provided, defaults will be used (see function
%                   m5pparams for details).
%   isBinCat      : A vector of flags indicating type of each input
%                   variable - either continuous (false) or categorical
%                   (true) with any number of categories, including binary.
%                   The vector should be of the same length as the number
%                   of columns in Xtr. m5pbuild then detects all the
%                   actually possible values for categorical variables from
%                   the training data. Any new values detected later, i.e.,
%                   during prediction, will be treated as NaN. By default,
%                   the vector is created with all values equal to false,
%                   meaning that all the variables are treated as
%                   continuous.
%   trainParamsEnsemble : A structure of parameters for building ensemble
%                   of trees. If not provided, a single tree is built. See
%                   function m5pparamsensemble for details. This can also
%                   be useful for variable importance assessment. See
%                   user's manual for examples of usage.
%                   Note that the ensemble building algorithm employs
%                   random number generator for which you can set seed
%                   before calling m5pbuild.
%   keepNodeInfo  : Whether to keep models (in model trees) and response
%                   values (in regression trees) in interior nodes of
%                   trees. And whether to keep indices of training
%                   observations that reached each node and standard
%                   deviation of each node. These are useful for further
%                   analysis and plotting. Default value = true. If set to
%                   false, the information is removed from the trees so
%                   that the structure takes up less memory. Note that
%                   interior nodes of smoothed trees will not contain
%                   models or response values regardless of the value of
%                   this parameter because only the models in the leaves
%                   are smoothed. Also note that the standard deviations
%                   are saved before doing smoothing.
%   verbose       : Whether to output additional information to console.
%                   (default value = true)
%
% Output:
%   model         : A single M5' tree (or a decision rule set) or a cell
%                   array of M5' trees (or decision rule sets) if an
%                   ensemble is built. A structure defining one tree (or a
%                   decision rule set) has the following fields:
%     binCat      : Information regarding original (continuous / binary /
%                   categorical) input variables, transformed (synthetic
%                   binary) input variables, possible values for
%                   categorical input variables and other information.
%     trainParams : A structure of training parameters for the algorithm
%                   (updated if any values are chosen automatically).
%     tree, rules, outcomes : Structures and arrays defining either the
%                   built tree or the generated decision rules.
%   time          : Algorithm execution time (in seconds).
%   ensembleResults : A structure of results for ensembles of trees or
%                   decision rules. Not available for Extra-Trees. The
%                   structure has the following fields:
%     OOBError    : Out-of-bag estimate of prediction Mean Squared Error of
%                   the ensemble after each new tree is built. The number
%                   of rows is equal to the number of trees built. OOBError
%                   is available only if getOOBError in trainParamsEnsemble
%                   is set to true. Note that it's possible to calculate
%                   mean out-of-bag predictions (and therefore out-of-bag
%                   errors for each individual training data observation)
%                   by summing the columns of OOBContrib.
%     OOBIndices  : Logical matrix. For each tree (column) indicates which
%                   observations were out-of-bag (and thus used in
%                   computing OOBError). The number of rows in OOBIndices
%                   is equal to the number of rows in Xtr and Ytr.
%                   OOBIndices is available only if getOOBError or
%                   getOOBContrib in trainParamsEnsemble is set to true.
%     OOBNum      : Number of times observations were out-of-bag (and thus
%                   used in computing OOBError). The length of OOBNum is
%                   equal to the number of rows in Xtr and Ytr. OOBNum is
%                   available only if getOOBError or getOOBContrib in
%                   trainParamsEnsemble is set to true.
%     OOBContrib  : A matrix allowing interpreting ensembles in accordance
%                   with the Forest Floor methodology (Welling et al.,
%                   2016). See also example of usage in Section 3.2 of
%                   user's manual.
%                   It is a matrix of contributions of each input variable
%                   to the response for each Xtr row in terms of response
%                   value changes along the prediction path of a tree
%                   (averaged over the whole ensemble) so that Yoob =
%                   in-bag_mean + x1_contribution + x2_contribution + ... +
%                   xn_contribution, where Yoob is prediction of response
%                   for out-of-bag observation. OOBContrib has the same
%                   number of columns as Xtr plus one, the last column
%                   being the in-bag response mean. The sum of columns of
%                   OOBContrib is equal to Yoob of the whole ensemble for
%                   each row of Xtr.
%                   OOBContrib is available only if getOOBContrib in
%                   trainParamsEnsemble is set to true.
%                   Note that it's also possible to compute contributions
%                   and explain predictions for new data (including with
%                   single trees) – see function m5ppredict.
%     varImportance : Variable importance assessment. Calculated when
%                   out-of-bag data of a variable is permuted. A matrix
%                   with four rows and as many columns as there are columns
%                   in Xtr. First row is the average increase of out-of-bag
%                   Mean Absolute Error (MAE), second row is standard
%                   deviation of the average increase of MAE, third row is
%                   the average increase of out-of-bag Mean Squared Error
%                   (MSE), fourth row is standard deviation of the average
%                   increase of MSE. The final variable importance estimate
%                   is often calculated by dividing each MAE or MSE by the
%                   corresponding standard deviation. Bigger values then
%                   indicate bigger importance of the corresponding
%                   variable. See user's manual for example of usage.
%                   varImportance is available only if getVarImportance in
%                   trainParamsEnsemble is > 0.

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

% Citing the M5PrimeLab toolbox:
% Jekabsons G., M5PrimeLab: M5' regression tree, model tree, and tree ensemble
% toolbox for Matlab/Octave, 2016, available at http://www.cs.rtu.lv/jekabsons/

% Last update: August 5, 2016

if nargin < 2
    error('Not enough input arguments.');
end

if isempty(Xtr) || isempty(Ytr)
    error('Training data is empty.');
end
if iscell(Xtr) || iscell(Ytr)
    error('Xtr and Ytr should not be cell arrays.');
end
[n, mOriginal] = size(Xtr); % number of observations and number of input variables
if size(Ytr,1) ~= n
    error('The number of rows in Xtr and Ytr should be equal.');
end
if size(Ytr,2) ~= 1
    error('Ytr should have one column.');
end
if any(any(isnan(Ytr)))
    error('Cannot handle NaNs in Ytr.');
end

if (nargin < 3) || isempty(trainParams)
    trainParams = m5pparams();
else
    trainParams.minLeafSize = max(1, trainParams.minLeafSize);
    if (trainParams.minLeafSize == 1) && trainParams.prune
        error('M5'' does not allow minLeafSize=1 if pruning is enabled.');
    end
    trainParams.minParentSize = max(trainParams.minLeafSize * 2, trainParams.minParentSize);
    if (trainParams.extractRules < 2)
        trainParams.smoothingK = max(0, trainParams.smoothingK);
    else
        if trainParams.smoothingK > 0
            warning('Smoothing for M5''Rules method is always disabled.');
        end
        trainParams.smoothingK = 0;
    end
    trainParams.splitThreshold = max(0, trainParams.splitThreshold);
    trainParams.maxDepth = max(0, floor(trainParams.maxDepth));
    trainParams.extractRules = max(0, min(2, floor(trainParams.extractRules)));
end
if (nargin < 4) || isempty(isBinCat)
    isBinCat = false(1,mOriginal);
else
    isBinCat = isBinCat(:)'; % force row vector
    if length(isBinCat) ~= mOriginal
        error('The number of elements in isBinCat should be equal to the number of columns in Xtr.');
    end
end
if (nargin < 5)
    trainParamsEnsemble = [];
else
    if (~isempty(trainParamsEnsemble)) && trainParamsEnsemble.extraTrees && (trainParams.prune || trainParams.modelTree)
        error('Pruning and model trees are not available for Extra-Trees.');
    end
end
if (nargin < 6) || isempty(keepNodeInfo)
    keepNodeInfo = true;
    if trainParams.smoothingK > 0
        keepInteriorModels = false; % forcing
    else
        keepInteriorModels = true;
    end
else
    keepInteriorModels = false;
    keepNodeInfo = false;
end
if (nargin < 7) || isempty(verbose)
    verbose = true;
end

binCat = isBinCat .* 2;
% Transform categorical variables into a number of synthetic binary variables
binCatVals = {};
if any(binCat >= 2)
    binCatNewNum = [];
    binCatCounter = 0;
    Xnew = [];
    model.binCat.varMap = {};
    for i = 1 : mOriginal
        if binCat(i) >= 2
            XX = Xtr(:,i);
            u = unique(XX(~isnan(XX))); % no NaNs, unique, sorted
            if size(u,1) > 2
                model.binCat.varMap = [model.binCat.varMap (size(binCatNewNum,2)+1) : (size(binCatNewNum,2)+size(u,1)-1)];
                avg = zeros(size(u,1),1);
                for j = 1 : size(u,1)
                    avg(j) = mean(Ytr(Xtr(:,i) == u(j)));
                end
                [~, ind] = sort(avg);
                u = u(ind);
                Xb = zeros(n,size(u,1)-1);
                for j = 1 : n
                    if isnan(Xtr(j,i))
                        Xb(j,:) = NaN;
                    else
                        Xb(j, 1 : find(Xtr(j,i) == u) - 1) = 1;
                    end
                end
                Xnew = [Xnew Xb];
                binCatNewNum = [binCatNewNum repmat(size(u,1),1,size(u,1)-1)];
            else
                Xnew = [Xnew Xtr(:,i)];
                binCatNewNum = [binCatNewNum 2];
                model.binCat.varMap = [model.binCat.varMap size(binCatNewNum,2)];
            end
            binCat(i) = size(u,1);
            binCatCounter = binCatCounter + 1;
            binCatVals{binCatCounter} = u;
            if binCat(i) >= 50
                warning(['Categorical variable #' num2str(i) ' has ' num2str(binCat(i)) ' unique values.']);
            end
        else
            Xnew = [Xnew Xtr(:,i)];
            binCatNewNum = [binCatNewNum 0];
            model.binCat.varMap = [model.binCat.varMap size(binCatNewNum,2)];
        end
    end
    Xtr = Xnew;
    model.binCat.catVals = binCatVals;
else
    binCatNewNum = binCat;
    model.binCat.varMap = num2cell(1:mOriginal);
end

model.binCat.binCat = binCat;
model.binCat.binCatNew = binCatNewNum >= 2; % 0 for continuous; 1 for binary
if any(model.binCat.binCatNew)
    % this is used later for printing/plotting of the trees/rules
    model.binCat.minVals = min(Xtr);
    if (trainParams.extractRules > 0)
        model.binCat.maxVals = max(Xtr);
    end
end

model.trainParams = trainParams;

if verbose
    if trainParams.modelTree, str = 'model'; else str = 'regression'; end
    if isempty(trainParamsEnsemble)
        if trainParams.extractRules == 0
            disp(['Growing M5'' ' str ' tree...']);
        else
            disp('Generating rule set...');
        end
    else
        if trainParams.extractRules == 0
            disp(['Growing M5'' ' str ' tree ensemble...']);
        else
            disp('Growing ensemble of rule sets...');
        end
    end
end

origWarningState = warning;
if exist('OCTAVE_VERSION', 'builtin')
    warning('off', 'Octave:nearly-singular-matrix');
    warning('off', 'Octave:singular-matrix');
else
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
end
ttt = tic;

% For the original binary and continuous variables beta = 1
% For synthetic binary variables created from original categorical variables beta < 1
beta = exp(7 * (2 - max(2, binCatNewNum)) / n);

if isempty(trainParamsEnsemble)
    
    sd = stdMy(Ytr);
    numNotMissing = sum(~isnan(Xtr),1); % number of non-missing values for each variable
    model = buildTree(model, Xtr, Ytr, sd, numNotMissing, binCatNewNum, beta, [], [], false, keepInteriorModels, keepNodeInfo);
    
    ensembleResults = [];
    
else
    
    if trainParamsEnsemble.numVarsTry < 1
        if trainParamsEnsemble.numVarsTry < 0
            trainParamsEnsemble.numVarsTry = mOriginal / 3;
        else
            trainParamsEnsemble.numVarsTry = mOriginal;
        end
    end
    trainParamsEnsemble.numVarsTry = min(mOriginal, max(1, floor(trainParamsEnsemble.numVarsTry)));
    
    if ~trainParamsEnsemble.extraTrees
        % Random Forests or Bagging
        
        if round(trainParamsEnsemble.inBagFraction * n) < 1
            error('trainParamsEnsemble.inBagFraction too small. In-bag set is empty.');
        end
        if (~trainParamsEnsemble.withReplacement) && (round(trainParamsEnsemble.inBagFraction * n) >= n)
            error('trainParamsEnsemble.inBagFraction too big. Out-of-bag set is empty.');
        end
        
        modelBase = model;
        models = cell(trainParamsEnsemble.numTrees, 1);
        
        if (~trainParamsEnsemble.getOOBError) && (trainParamsEnsemble.getVarImportance == 0) && (~trainParamsEnsemble.getOOBContrib)
            ensembleResults = [];
        else
            if trainParamsEnsemble.getOOBError || trainParamsEnsemble.getOOBContrib
                OOBNum = zeros(n, 1);
                ensembleResults.OOBIndices = false(n, trainParamsEnsemble.numTrees);
            end
            if trainParamsEnsemble.getOOBContrib
                OOBContrib = zeros(n, mOriginal + 1);
            end
            if trainParamsEnsemble.getOOBError
                OOBPred = zeros(n, 1);
                ensembleResults.OOBError = NaN(trainParamsEnsemble.numTrees, 1);
            end
            if trainParamsEnsemble.getVarImportance > 0
                diffOOBMAE = NaN(trainParamsEnsemble.numTrees, mOriginal);
                diffOOBMSE = NaN(trainParamsEnsemble.numTrees, mOriginal);
                ensembleResults.varImportance = zeros(4, mOriginal); % increase in MAE, SD, increase in MSE, SD
            end
        end
        
        % for each tree
        for t = 1 : trainParamsEnsemble.numTrees
            if verbose && (trainParamsEnsemble.verboseNumIter > 0) && ...
                    (mod(t, trainParamsEnsemble.verboseNumIter) == 0)
                if trainParams.extractRules == 0
                    fprintf('Growing tree #%d...\n', t);
                else
                    fprintf('Generating rule set #%d...\n', t);
                end
            end
            % sampling
            if trainParamsEnsemble.withReplacement
                idx = randi(n, round(trainParamsEnsemble.inBagFraction * n), 1);
                X = Xtr(idx,:);
                Y = Ytr(idx,1);
            else
                perm = randperm(n);
                idx = perm(1:round(trainParamsEnsemble.inBagFraction * n));
                X = Xtr(idx,:);
                Y = Ytr(idx,1);
            end
            
            if t > 1
                model = modelBase;
            end
            
            sd = stdMy(Y);
            numNotMissing = sum(~isnan(X),1); % number of non-missing values for each variable
            model = buildTree(model, X, Y, sd, numNotMissing, binCatNewNum, beta, ...
                trainParamsEnsemble.numVarsTry, mOriginal, false, keepInteriorModels, keepNodeInfo);
            
            % additional calculations, if asked
            if trainParamsEnsemble.getOOBError || (trainParamsEnsemble.getVarImportance > 0) || trainParamsEnsemble.getOOBContrib
                idxoob = true(n,1);
                idxoob(idx) = false;
                idxoob = find(idxoob);
                if ~isempty(idxoob) % test for the unlikely case when out-of-bag set is empty
                    Xoob = Xtr(idxoob,:);
                    Yq = zeros(size(Xoob,1),1);
                    
                    if trainParamsEnsemble.getOOBContrib
                        for i = 1 : size(Xoob,1)
                            [Yq(i), OOBContrib2] = predictsingle(model, Xoob(i,:), modelBase.binCat.varMap);
                            OOBContrib(idxoob(i),:) = OOBContrib(idxoob(i),:) + OOBContrib2;
                        end
                    else
                        for i = 1 : size(Xoob,1)
                            Yq(i) = predictsingle(model, Xoob(i,:));
                        end
                    end
                    
                    if trainParamsEnsemble.getOOBError || trainParamsEnsemble.getOOBContrib
                        ensembleResults.OOBIndices(idxoob,t) = true;
                        OOBNum(idxoob) = OOBNum(idxoob) + 1;
                    end
                    
                    if trainParamsEnsemble.getOOBError
                        idxExist = OOBNum ~= 0;
                        OOBPred(idxoob) = OOBPred(idxoob) + Yq;
                        ensembleResults.OOBError(t,1) = mean(((OOBPred(idxExist) ./ OOBNum(idxExist)) - Ytr(idxExist)) .^ 2);
                        if verbose && (trainParamsEnsemble.verboseNumIter > 0) && ...
                                (mod(t, trainParamsEnsemble.verboseNumIter) == 0)
                            if trainParams.extractRules == 0
                                fprintf('Out-of-bag MSE with %d trees: %.5g\n', t, ensembleResults.OOBError(t,1));
                            else
                                fprintf('Out-of-bag MSE with %d rule sets: %.5g\n', t, ensembleResults.OOBError(t,1));
                            end
                        end
                    end
                    
                    if trainParamsEnsemble.getVarImportance > 0
                        Yqtdiff = Yq - Ytr(idxoob);
                        for v = 1 : mOriginal
                            for iPerm = 1:trainParamsEnsemble.getVarImportance
                                Xoobpert = Xoob;
                                idxoobpert = idxoob(randperm(size(idxoob,1)),1);
                                % Perturb OOB variables that correspond to the original vth variable
                                for vnew = model.binCat.varMap{v}
                                    Xoobpert(:,vnew) = Xtr(idxoobpert,vnew);
                                end
                                Yqpert = zeros(size(Xoobpert,1),1);
                                for i = 1 : size(Xoobpert,1)
                                    Yqpert(i) = predictsingle(model, Xoobpert(i,:));
                                end
                                Yqptdiff = Yqpert - Ytr(idxoob);
                                if iPerm == 1
                                    diffOOBMAE(t,v) = mean(abs(Yqptdiff)) - mean(abs(Yqtdiff));
                                    diffOOBMSE(t,v) = mean(Yqptdiff .^ 2) - mean(Yqtdiff .^ 2);
                                else
                                    diffOOBMAE(t,v) = diffOOBMAE(t,v) + mean(abs(Yqptdiff)) - mean(abs(Yqtdiff));
                                    diffOOBMSE(t,v) = diffOOBMSE(t,v) + mean(Yqptdiff .^ 2) - mean(Yqtdiff .^ 2);
                                end
                            end
                            if trainParamsEnsemble.getVarImportance > 1
                                diffOOBMAE(t,v) = diffOOBMAE(t,v) / trainParamsEnsemble.getVarImportance;
                                diffOOBMSE(t,v) = diffOOBMSE(t,v) / trainParamsEnsemble.getVarImportance;
                            end
                        end
                    end

                end
            end
            
            models{t} = model;
        end % end of loop through all trees
        model = models;
        if trainParamsEnsemble.getOOBError || trainParamsEnsemble.getOOBContrib
            ensembleResults.OOBNum = OOBNum;
        end
        if trainParamsEnsemble.getOOBContrib
            ensembleResults.OOBContrib = OOBContrib ./ repmat(OOBNum, 1, mOriginal + 1);
        end
        if trainParamsEnsemble.getVarImportance > 0
            ensembleResults.varImportance(1,:) = mean(diffOOBMAE, 1);
            ensembleResults.varImportance(2,:) = std(diffOOBMAE, 1, 1);
            ensembleResults.varImportance(3,:) = mean(diffOOBMSE, 1);
            ensembleResults.varImportance(4,:) = std(diffOOBMSE, 1, 1);
        end
        
    else % if extraTrees
        % Extra-Trees
        
        modelBase = model;
        models = cell(trainParamsEnsemble.numTrees, 1);
        ensembleResults = [];
        sd = stdMy(Ytr);
        numNotMissing = sum(~isnan(Xtr),1); % number of non-missing values for each variable
        % for each tree
        for t = 1 : trainParamsEnsemble.numTrees
            if verbose && (trainParamsEnsemble.verboseNumIter > 0) && ...
                    (mod(t, trainParamsEnsemble.verboseNumIter) == 0)
                if trainParams.extractRules == 0
                    fprintf('Growing tree #%d...\n', t);
                else
                    fprintf('Generating rule set #%d...\n', t);
                end
            end
            if t > 1
                model = modelBase;
            end
            model = buildTree(model, Xtr, Ytr, sd, numNotMissing, binCatNewNum, beta, ...
                trainParamsEnsemble.numVarsTry, mOriginal, true, keepInteriorModels, keepNodeInfo);
            
            models{t} = model;
        end
        model = models;
    end % end of if extraTrees
    
end

time = toc(ttt);
if verbose
    if isempty(trainParamsEnsemble)
        printinfo(model);
    end
    fprintf('Execution time: %0.2f seconds\n', time);
end
warning(origWarningState);
end

%==========================================================================

function model = buildTree(model, X, Y, sd, numNotMissing, binCatNewNum, beta, numVarsTry, mOriginal, extraTrees, keepInteriorModels, keepNodeInfo)
% Builds a tree. If asked, extracts decision rules and returns them instead of the tree.
if model.trainParams.extractRules == 0
    % This is normal execution for building M5' trees.
    % Growing the tree
    model.tree = splitNode(X, Y, 1:size(Y,1), 0, sd, numNotMissing, binCatNewNum, model.trainParams, beta, numVarsTry, mOriginal, model.binCat.varMap, extraTrees, keepNodeInfo);
    % Pruning the tree and/or filling it with models or mean values
    model.tree = pruneNode(model.tree, X, Y, model.trainParams, keepNodeInfo);
    if model.trainParams.smoothingK > 0
        totalAttrs = model.binCat.varMap{end}(end);
        model.tree = smoothing(model.tree, [], model.trainParams.modelTree, model.trainParams.smoothingK, totalAttrs);
    end
    model.tree = cleanUp(model.tree, model.trainParams.modelTree, ~keepInteriorModels, ~keepNodeInfo);
elseif model.trainParams.extractRules == 1
    % Build M5' tree and extract all its decision rules.
    % Growing the tree
    tree = splitNode(X, Y, 1:size(Y,1), 0, sd, numNotMissing, binCatNewNum, model.trainParams, beta, numVarsTry, mOriginal, model.binCat.varMap, extraTrees, keepNodeInfo);
    % Pruning the tree and/or filling it with models or mean values
    tree = pruneNode(tree, X, Y, model.trainParams, keepNodeInfo);
    if model.trainParams.smoothingK > 0
        totalAttrs = model.binCat.varMap{end}(end);
        tree = smoothing(tree, [], model.trainParams.modelTree, model.trainParams.smoothingK, totalAttrs);
    end
    if model.trainParams.modelTree
        [model.rules, model.outcomesCoefs, model.outcomesAttrIdx, model.outcomesAttrAvg, model.outcomesNumCases, outcomesCaseIdx, outcomesSD] = ...
            createRules(tree, model.trainParams.modelTree, false, keepNodeInfo);
    else
        [model.rules, model.outcomes, ~, ~, model.outcomesNumCases, outcomesCaseIdx, outcomesSD] = ...
            createRules(tree, model.trainParams.modelTree, false, keepNodeInfo);
    end
    if keepNodeInfo
        model.outcomesCaseIdx = outcomesCaseIdx;
        model.outcomesSD = outcomesSD;
    end
else
    % Builds a list of decision rules using the M5'Rules method.
    model.rules = {};
    if model.trainParams.modelTree
        model.outcomesCoefs = {};
        model.outcomesAttrIdx = {};
        model.outcomesAttrAvg = {};
    else
        model.outcomes = [];
    end
    model.outcomesNumCases = [];
    if keepNodeInfo
        caseIdx = 1 : size(X,1); % so that we get original indices even after some observations are deleted
        model.outcomesCaseIdx = {};
        model.outcomesSD = [];
    end
    currRule = 0;
    while true
        % Growing the tree
        tree = splitNode(X, Y, 1:size(Y,1), 0, sd, numNotMissing, binCatNewNum, model.trainParams, beta, numVarsTry, mOriginal, model.binCat.varMap, extraTrees, keepNodeInfo);
        % Pruning the tree and/or filling it with models or mean values
        tree = pruneNode(tree, X, Y, model.trainParams, keepNodeInfo);
        if model.trainParams.modelTree
            [rules, outcomesCoefs, outcomesAttrIdx, outcomesAttrAvg, outcomesNumCases, outcomesCaseIdx, outcomesSD] = ...
                createRules(tree, model.trainParams.modelTree, true, keepNodeInfo);
        else
            [rules, outcomes, ~, ~, outcomesNumCases, outcomesCaseIdx, outcomesSD] = ...
                createRules(tree, model.trainParams.modelTree, true, keepNodeInfo);
        end
        [~, idx] = max(outcomesNumCases);
        
        % Storing the decision rule with the biggest coverage.
        currRule = currRule + 1;
        model.rules{currRule,1} = rules{idx};
        if model.trainParams.modelTree
            model.outcomesCoefs{currRule,1} = outcomesCoefs{idx};
            model.outcomesAttrIdx{currRule,1} = outcomesAttrIdx{idx};
            model.outcomesAttrAvg{currRule,1} = outcomesAttrAvg{idx};
        else
            model.outcomes(currRule,1) = outcomes(idx);
        end
        model.outcomesNumCases(currRule,1) = outcomesNumCases(idx);
        if keepNodeInfo
            model.outcomesCaseIdx{currRule,1} = caseIdx(outcomesCaseIdx{idx});
            caseIdx(outcomesCaseIdx{idx}) = [];
            model.outcomesSD(currRule,1) = outcomesSD(idx);
        end
        
        % Deleting observations covered by the stored rule.
        X(outcomesCaseIdx{idx},:) = [];
        Y(outcomesCaseIdx{idx},:) = [];
        if (size(X,1) == 0)
            break;
        end
        %sd = stdMy(Y);
        %numNotMissing = sum(~isnan(X),1);
    end
end
end

function [node, attrList] = splitNode(X, Y, caseIdx, depth, sd, numNotMissing, binCatNewNum, trainParams, beta, numVarsTry, mOriginal, varMap, extraTrees, keepNodeInfo)
% The function splits node into left node and right node
node.caseIdx = caseIdx;
if depth >= trainParams.maxDepth
    node.interior = false; % this node will be a leaf node
    attrList = [];
    if keepNodeInfo
        node.sd = stdMy(Y(caseIdx));
    end
    return;
end
YY = Y(caseIdx);
stdYall = stdMy(YY);
if keepNodeInfo
    node.sd = stdYall;
end
% no need to check minLeafSize*2 because minParentSize is guaranteed to be at least twice the minLeafSize
% (size(caseIdx,2) < trainParams.minLeafSize * 2) || ...
if (size(caseIdx,2) < trainParams.minParentSize) || ...
   (stdYall < trainParams.splitThreshold * sd)
    node.interior = false; % this node will be a leaf node
    attrList = [];
    return;
end;
sdr = -Inf;
if ~extraTrees
    % This is for individual trees and trees in Bagging and Random Forests
    if isempty(numVarsTry) || (numVarsTry >= mOriginal)
        varsTry = 1:size(X, 2); % try all variables
    else
        % We will try random subset of variables (for building ensembles)
        % For categorical variables, we will try all their synthetic binary variables
        origVList = randperm(mOriginal);
        varsTry = [varMap{origVList(1:numVarsTry)}];
    end
else
    % This is for trees in Extra-Trees
    if isempty(numVarsTry) || (numVarsTry >= mOriginal)
        % This is for the typical configuration when we try all variables, for one split each
        varsTry = [];
        for v = 1 : mOriginal
            vars = varMap{v};
            if size(vars,2) < 2
                varsTry = [varsTry vars];
            else
                % For categorical variables, we will randomly select one synthetic binary variable
                varsTry = [varsTry vars(randi(size(vars,2),1))];
            end
        end
    else
        % This is for the configuration when we try fewer than all variables but they should be non constant in the node
        origVList = randperm(mOriginal);
        numVarsUsed = 0;
        varsTry = [];
        for origV = origVList
            vars = varMap{origV};
            nonConstant = false;
            for v = vars
                XX = X(caseIdx,v);
                if min(XX) ~= max(XX)
                    nonConstant = true;
                    break;
                end
            end
            if ~nonConstant
                continue;
            end
            if size(vars,2) >= 2
                % For categorical variables, we will randomly select one synthetic binary variable
                vars = vars(randi(size(vars,2),1));
            end
            varsTry = [varsTry vars];
            numVarsUsed = numVarsUsed + 1;
            if numVarsUsed >= numVarsTry
                break;
            end
        end
    end
end
% let's find best variable and best split
for i = varsTry
    XX = X(caseIdx,i);
    % NaNs (unknown values) will not be used for split point determination
    % and there is no need to sort because unique already sorts
    nonansIdx = ~isnan(XX);
    XXnonans = XX(nonansIdx);
    if binCatNewNum(i) >= 2
        % It's simple with binary variables
        minXXnonans = min(XXnonans);
        maxXXnonans = max(XXnonans);
        if minXXnonans == maxXXnonans
            continue;
        end
        splitCandidates = (minXXnonans + maxXXnonans) / 2;
    else
        if ~extraTrees
            sorted = unique(XXnonans);
            if size(sorted,1) < 2
                continue;
            end
            splitCandidates = ((sorted(1:end-1) + sorted(2:end)) ./ 2)';
        else
            minXXnonans = min(XXnonans);
            maxXXnonans = max(XXnonans);
            if minXXnonans == maxXXnonans
                continue;
            end
            splitCandidates = minXXnonans + rand(1) * (maxXXnonans - minXXnonans);
        end
    end
    sizeAllNoNans = size(XXnonans,1); % size without NaNs
    if (sizeAllNoNans == size(XX,1)) % if there are no NaNs
        stdY = stdYall;
    else
        stdY = stdMy(YY(nonansIdx)); % NaNs are not used for splitting decisions
    end
    % let's find the best split
    for splitCand = splitCandidates
        leftInd = find(XX <= splitCand);
        if (size(leftInd,1) < trainParams.minLeafSize)
            continue;
        end
        rightInd = find(XX > splitCand);
        if (size(rightInd,1) < trainParams.minLeafSize)
            break; % break loop because we definitely are too near the edge for any further split point to be allowed
        end
        % calculate SDR for the split point
        if trainParams.vanillaSDR
            sdrNew = stdY - (size(leftInd,1) * stdMy(YY(leftInd)) + size(rightInd,1) * stdMy(YY(rightInd))) / sizeAllNoNans;
        else
            sdrNew = numNotMissing(i) / sizeAllNoNans * beta(i) * ...
                (stdY - (size(leftInd,1) * stdMy(YY(leftInd)) + size(rightInd,1) * stdMy(YY(rightInd))) / sizeAllNoNans);
        end
        if sdrNew > sdr
            sdr = sdrNew;
            splitPoint = splitCand;
            attrList = i;
        end
    end
end
if sdr <= 0
    % This node will be a leaf node
    node.interior = false;
    attrList = [];
else
    % This node will be an interior node
    [leftInd, rightInd] = leftright(splitPoint, X(caseIdx,attrList), YY, binCatNewNum(attrList));
    leftInd = caseIdx(leftInd);
    rightInd = caseIdx(rightInd);
    node.interior = true;
    node.splitAttr = attrList;
    node.splitLocation = splitPoint;
    [node.left, attrList2] = ...
        splitNode(X, Y, leftInd, depth + 1, sd, numNotMissing, binCatNewNum, trainParams, ...
                beta, numVarsTry, mOriginal, varMap, extraTrees, keepNodeInfo);
    if trainParams.modelTree
        attrList = [attrList attrList2];
    end
    [node.right, attrList2] = ...
        splitNode(X, Y, rightInd, depth + 1, sd, numNotMissing, binCatNewNum, trainParams, ...
                beta, numVarsTry, mOriginal, varMap, extraTrees, keepNodeInfo);
    if trainParams.modelTree
        attrList = unique([attrList attrList2]); % unique also sorts
        node.attrList = attrList;
    end
end
end

function stdev = stdMy(Y)
% Calculates standard deviation
% Does the same as Matlab's std function but without all the overhead
nn = size(Y,1);
stdev = sqrt(sum((Y - (sum(Y) / nn)) .^ 2) / nn);
end

function [leftInd, rightInd] = leftright(split, X, Y, binCatNewNum)
% Splits all observations into left and right sets. Deals with NaNs separately.
leftInd = find(X <= split);
rightInd = find(X > split);
% Place observations with NaNs in left or right according to their Y values
isNaN = isnan(X);
if any(isNaN)
    if binCatNewNum < 2
        % For continuous variables
        [~, sorted] = sort(X(leftInd));
        sorted = leftInd(sorted);
        leftAvg = mean(Y(sorted(end - min([2 size(leftInd,1)-1]) : end)));
        [~, sorted] = sort(X(rightInd));
        sorted = rightInd(sorted);
        rightAvg = mean(Y(sorted(1 : min([3 size(rightInd,1)]))));
    else
        % For both original and synthetic binary variables
        leftAvg = mean(Y(leftInd));
        rightAvg = mean(Y(rightInd));
    end
    avgAvg = (leftAvg + rightAvg) / 2;
    smaller = Y(isNaN) <= avgAvg;
    nanInd = find(isNaN);
    if leftAvg <= rightAvg
        leftInd = [leftInd; nanInd(smaller)];
        rightInd = [rightInd; nanInd(~smaller)];
    else
        leftInd = [leftInd; nanInd(~smaller)];
        rightInd = [rightInd; nanInd(smaller)];
    end
end
end

function node = pruneNode(node, X, Y, trainParams, keepNodeInfo)
% Prunes the tree and fills it with models (or average values).
% If tree pruning is disabled, only filling with models is done.
% For each model, subset selection is done (using sequential backward selection).
if ~node.interior
    if ~trainParams.modelTree
        node.value = mean(Y(node.caseIdx));
    else
        % Original leaf nodes ignore input variables
        node.modelCoefs = mean(Y(node.caseIdx));
        node.modelAttrIdx = [];
    end
    return;
end
node.left = pruneNode(node.left, X, Y, trainParams, keepNodeInfo);
node.right = pruneNode(node.right, X, Y, trainParams, keepNodeInfo);
if ~trainParams.modelTree
    node.value = mean(Y(node.caseIdx));
    if trainParams.prune
        errNode = calcErrNodeWithAllKnown(node, X, Y, trainParams, true); % pretend it's known because regression tree doesn't care
    end
else
    attrInd = node.attrList;
    if isempty(attrInd) % no attributes. the model will include only intercept
        node.modelCoefs = mean(Y(node.caseIdx));
        node.modelAttrIdx = [];
        if trainParams.prune
            errNode = calcErrNodeWithAllKnown(node, X, Y, trainParams, true); % pretend it's known because no attributes are used
        end
    else
        XX = X;
        isNaN = isnan(X(node.caseIdx,attrInd));
        for i = 1 : length(attrInd)
            % Store average values of the variables (required when the tree
            % is used for prediction and NaN is encountered)
            % (node.modelAttrIdx provides index for the variable for which
            % modelAttrAvg is the average value)
            node.modelAttrAvg(i) = mean(X(node.caseIdx(~isNaN(:,i)),attrInd(i)));
            % Replace NaNs by the average values of the corresponding variables
            % of the training observations reaching the node
            XX(node.caseIdx(isNaN(:,i)),attrInd(i)) = node.modelAttrAvg(i);
        end
        A = [ones(length(node.caseIdx),1) XX(node.caseIdx,attrInd)];
        node.modelCoefs = A \ Y(node.caseIdx);
        node.modelAttrIdx = attrInd;
        if trainParams.prune
            errNode = calcErrNodeWithAllKnown(node, XX, Y, trainParams, true);
            if trainParams.eliminateTerms
                % Perform variable selection
                attrIndBest = attrInd;
                coefsBest = node.modelCoefs;
                changed = false;
                for j = 1 : length(attrInd)
                    attrIndOld = node.modelAttrIdx;
                    for i = 1 : length(attrIndOld)
                        node.modelAttrIdx = attrIndOld;
                        node.modelAttrIdx(i) = [];
                        A = [ones(length(node.caseIdx),1) XX(node.caseIdx,node.modelAttrIdx)];
                        node.modelCoefs = A \ Y(node.caseIdx);
                        errTry = calcErrNodeWithAllKnown(node, XX, Y, trainParams, true);
                        if errTry < errNode
                            errNode = errTry;
                            attrIndBest = node.modelAttrIdx;
                            coefsBest = node.modelCoefs;
                            changed = true;
                        end
                    end
                    node.modelAttrIdx = attrIndBest;
                    node.modelCoefs = coefsBest;
                    if ~changed
                        break;
                    end
                end
                % Update node.modelAttrAvg if the used subset of variables has changed
                if length(node.modelAttrIdx) < length(attrInd)
                    for i = 1 : length(node.modelAttrIdx)
                        node.modelAttrAvg(i) = node.modelAttrAvg(attrInd == node.modelAttrIdx(i));
                    end
                    node.modelAttrAvg = node.modelAttrAvg(1:length(node.modelAttrIdx));
                end
            end
        end
        if keepNodeInfo
            val = [ones(length(node.caseIdx),1) X(node.caseIdx,node.modelAttrIdx)] * node.modelCoefs;
            node.sd = sqrt(mean((val - Y(node.caseIdx)).^2));
        end
    end
end
if trainParams.prune && ...
   ( ...
    ((~trainParams.aggressivePruning) && (calcErrSubtree(node, X, Y, trainParams) >= errNode)) || ...
    (trainParams.aggressivePruning && (calcErrSubtreeAggressive(node, X, Y, trainParams) >= errNode)) ...
   )
    % above we could also add "(sd * 1E-6 > errNode)"
    % this node will be a leaf node
    node.interior = false;
    if trainParams.modelTree
        node = rmfield(node, {'splitAttr', 'splitLocation', 'left', 'right', 'attrList'});
    else
        node = rmfield(node, {'splitAttr', 'splitLocation', 'left', 'right'});
    end
else
    if trainParams.modelTree
        node = rmfield(node, 'attrList');
    end
    % Store average value of the split variable (required when the tree
    % is used for prediction and NaN is encountered)
    notNaN = node.caseIdx(~isnan(X(node.caseIdx,node.splitAttr)));
    %node.splitAttrAvg = mean(X(notNaN,node.splitAttr)); % not really needed. we can just set nanLeft
    node.nanLeft = mean(X(notNaN,node.splitAttr)) <= node.splitLocation;
end
end

function err = calcErrSubtree(node, X, Y, trainParams)
% Calculates error of the subtree
if node.interior
    err = (length(node.left.caseIdx) * calcErrSubtree(node.left, X, Y, trainParams) + ...
           length(node.right.caseIdx) * calcErrSubtree(node.right, X, Y, trainParams)) / ...
           length(node.caseIdx);
else
    err = calcErrNode(node, X, Y, trainParams);
end
end

function err = calcErrSubtreeAggressive(node, X, Y, trainParams)
% Calculates error of the subtree, applies penalty
[err, v] = calcErrSubtreeAggressiveDo(node, X, Y, trainParams);
nn = length(node.caseIdx);
if (nn > v)
    err = err * (nn + v * 2) / (nn - v);
else
    err = err * 10;
end
end
function [err, v] = calcErrSubtreeAggressiveDo(node, X, Y, trainParams)
% Calculates error of the subtree
if node.interior
    [errLeft, vLeft] = calcErrSubtreeAggressiveDo(node.left, X, Y, trainParams);
    [errRight, vRight] = calcErrSubtreeAggressiveDo(node.right, X, Y, trainParams);
    err = (length(node.left.caseIdx) * errLeft + length(node.right.caseIdx) * errRight) / length(node.caseIdx);
    v = vLeft + vRight + 1;
else
    err = calcErrNode(node, X, Y, trainParams);
    if trainParams.modelTree
        v = length(node.modelCoefs);
    else
        v = 1;
    end
end
end

function err = calcErrNode(node, X, Y, trainParams)
% Calculates error of the node. Handles missing values.
if trainParams.modelTree
    % Replace NaNs with the average values of the corresponding variables
    % of the training observations reaching the node
    isNaN = isnan(X(node.caseIdx,node.modelAttrIdx));
    for i = 1 : length(node.modelAttrIdx)
        X(node.caseIdx(isNaN(:,i)),node.modelAttrIdx(i)) = node.modelAttrAvg(i);
    end
end
err = calcErrNodeWithAllKnown(node, X, Y, trainParams, false);
end

function err = calcErrNodeWithAllKnown(node, X, Y, trainParams, forDroppingTerms)
% Calculates error of the node. Assumes all values are known.
if trainParams.modelTree
    val = [ones(length(node.caseIdx),1) X(node.caseIdx,node.modelAttrIdx)] * node.modelCoefs;
    deviation = mean(abs(val - Y(node.caseIdx)));
    v = length(node.modelCoefs);
else
    deviation = mean(abs(node.value - Y(node.caseIdx)));
    v = 1;
end
if ~trainParams.aggressivePruning
    nn = length(node.caseIdx);
    err = (nn + v) / (nn - v) * deviation;
else
    if forDroppingTerms
        nn = length(node.caseIdx);
        err = (nn + v * 2) / (nn - v) * deviation;
    else
        err = deviation;
    end
end
end

function node = cleanUp(node, modelTree, removeInteriorModels, removeCaseIdx)
% Removing the temporary fields
node.numCases = length(node.caseIdx);
if removeCaseIdx
    node = rmfield(node, 'caseIdx');
end
if node.interior
    if removeInteriorModels
        if modelTree
            node = rmfield(node, {'modelCoefs', 'modelAttrAvg', 'modelAttrIdx'});
        else
            node = rmfield(node, 'value');
        end
    end
    node.left = cleanUp(node.left, modelTree, removeInteriorModels, removeCaseIdx);
    node.right = cleanUp(node.right, modelTree, removeInteriorModels, removeCaseIdx);
end
end

function node = smoothing(node, list, modelTree, smoothingK, totalAttrs)
% Performs smoothing by incorporating interior models into leaf models.
% Deals with modelAttrAvg, so that unknown values can be substituted with
% modelAttrAvg at leaves.
if node.interior
    if modelTree
        data.attrIdx = node.modelAttrIdx;
        data.coefs = node.modelCoefs;
        data.attrAvg = zeros(totalAttrs,1);
        data.attrAvg(node.modelAttrIdx) = node.modelAttrAvg;
    else
        data.value = node.value;
    end
    data.numCases = length(node.caseIdx);
    list{end+1} = data; % making a list. will be used at leaf nodes
    node.left = smoothing(node.left, list, modelTree, smoothingK, totalAttrs);
    node.right = smoothing(node.right, list, modelTree, smoothingK, totalAttrs);
else
    if modelTree
        len = length(list);
        if len > 0
            attrIdx = node.modelAttrIdx;
            s_n = length(node.caseIdx);
            coefs = zeros(totalAttrs+1,1);
            coefs([1 attrIdx+1]) = node.modelCoefs;
            attrAvg = zeros(totalAttrs,1);
            if ~isempty(attrIdx)
                attrAvg(attrIdx) = node.modelAttrAvg;
            end
            % pretend to go from the leaf node to the root node
            for i = len:-1:1
                % Update list of used variables
                attrIdx = union(attrIdx, list{i}.attrIdx); % union sorts. this also will make equations easier to understand
                % Coefs at this node
                coefsHere = zeros(size(coefs));
                coefsHere([1 list{i}.attrIdx+1]) = list{i}.coefs;
                % Recalculate weighted averages for NaNs
                idx = true(size(coefs));
                idx(1) = false;
                idx((coefs == 0) & (coefsHere == 0)) = false;
                if any(idx)
                    idxAttr = idx(2:end);
                    attrAvg(idxAttr) = ...
                        attrAvg(idxAttr) .* s_n .* coefs(idx) ./ (s_n .* coefs(idx) + smoothingK .* coefsHere(idx)) + ...
                        list{i}.attrAvg(idxAttr) .* smoothingK .* coefsHere(idx) ./ (s_n .* coefs(idx) + smoothingK .* coefsHere(idx));
                end
                % Recalculate smoothed coefs
                coefs = (s_n * coefs + smoothingK * coefsHere) / (s_n + smoothingK);
                s_n = list{i}.numCases; % s_n for next iteration
            end
            attrIdx = attrIdx(:)'; % force row vector
            node.modelCoefs = coefs([1 attrIdx+1]);
            node.modelAttrIdx = attrIdx;
            node.modelAttrAvg = attrAvg(attrIdx)';
        end
    else
        len = length(list);
        if len > 0
            value = node.value;
            s_n = length(node.caseIdx);
            % pretend to go from the leaf node to the root node
            for i = len:-1:1
                value = (s_n * value + smoothingK * list{i}.value) / (s_n + smoothingK); % calculate smoothed values
                s_n = list{i}.numCases; % s_n for next iteration
            end
            node.value = value;
        end
    end
end
end

function [rules, outcomes, outcomesAttrIdx, outcomesAttrAvg, outcomesNumCases, outcomesCaseIdx, outcomesSD] = ...
        createRules(tree, modelTree, maxCoverageOnly, keepNodeInfo)
% Extracts decision rules from a tree.
totalRules = countRules(tree);
rules = cell(totalRules,1);
if modelTree
    outcomes = cell(totalRules,1);
    outcomesAttrIdx = cell(totalRules,1);
    outcomesAttrAvg = cell(totalRules,1);
else
    outcomes = zeros(totalRules,1);
    outcomesAttrIdx = [];
    outcomesAttrAvg = [];
end
outcomesNumCases = zeros(totalRules,1);
outcomesCaseIdx = cell(totalRules,1);
if keepNodeInfo
    outcomesSD = nan(totalRules,1);
else
    outcomesSD = [];
end
currRule = 0;
maxNumCases = 0;
createRulesDo(tree, {});
function createRulesDo(node, tests)
    if node.interior
        % Interior nodes become tests in rules
        tests{end+1,1}.attr = node.splitAttr;
        tests{end}.location = node.splitLocation;
        tests{end}.le = true; % "<="
        tests{end}.orNan = node.nanLeft; % whether to accept NaN
        createRulesDo(node.left, tests);
        tests{end}.le = false; % ">"
        tests{end}.orNan = ~node.nanLeft; % whether to accept NaN
        createRulesDo(node.right, tests);
        return;
    end
    % Leaf nodes become outcomes for the rules
    currRule = currRule + 1;
    if maxCoverageOnly && (length(node.caseIdx) <= maxNumCases)
        % If we'll actually want only the rule with the maximum coverage
        % then we don't need to store everything for rules that are already
        % known to be smaller.
        outcomesNumCases(currRule,1) = 0;
        return;
    end
    rules{currRule,1} = tests; % store all tests
    if modelTree
        outcomes{currRule,1} = node.modelCoefs;
        outcomesAttrIdx{currRule,1} = node.modelAttrIdx;
        if isempty(node.modelAttrIdx)
            outcomesAttrAvg{currRule,1} = [];
        else
            outcomesAttrAvg{currRule,1} = node.modelAttrAvg;
        end
    else
        outcomes(currRule,1) = node.value;
    end
    maxNumCases = length(node.caseIdx);
    outcomesNumCases(currRule,1) = maxNumCases;
    outcomesCaseIdx{currRule,1} = node.caseIdx;
    if keepNodeInfo
        outcomesSD(currRule,1) = node.sd;
    end
end
end

function nRules = countRules(node)
% Counts all rules (equal to the number of leaf nodes) in the tree.
if node.interior
    nRules = countRules(node.left) + countRules(node.right);
else
    nRules = 1;
end
end
