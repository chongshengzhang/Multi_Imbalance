function m5pprint(model, showNumCases, precision, dealWithNaN)
% m5pprint
% Prints M5' tree or decision rule set in a human-readable form. Does not
% work with ensembles.
%
% Call:
%   m5pprint(model, showNumCases, precision, dealWithNaN)
%
% All the input arguments, except the first one, are optional. Empty values
% are also accepted (the corresponding defaults will be used).
%
% Input:
%   model         : M5' model or decision rule set.
%   showNumCases  : Whether to show the number of training observations
%                   corresponding to each leaf (default value = true).
%   precision     : Number of digits used for any numerical values shown
%                   (default value = 5).
%   dealWithNaN   : Whether to display how the tree deals with missing
%                   values (NaN, displayed as '?') (default value =
%                   false).
%
% Remarks:
% 1. For smoothed M5' trees / decision rule sets, the smoothing process is
%    already done in m5pbuild, therefore if you want to see unsmoothed
%    versions (which are usually easier to interpret) you should build
%    trees with smoothing disabled.
% 2. If the training data has categorical variables with more than two
%    categories, the corresponding synthetic binary variables are shown.

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

% Last update: July 28, 2016

if nargin < 1
    error('Not enough input arguments.');
end
if length(model) > 1
    error('This function works with single trees only.');
else
    if iscell(model)
        model = model{1};
    end
end
if (nargin < 2) || isempty(showNumCases)
    showNumCases = true;
end
if (nargin < 3) || isempty(precision)
    precision = 5;
end
if (nargin < 4) || isempty(dealWithNaN)
    dealWithNaN = false;
end

% Print synthetic variables
if any(model.binCat.binCat > 2)
    disp('Synthetic variables:');
    indCounter = 0;
    binCatCounter = 0;
    for i = 1 : length(model.binCat.binCat)
        if model.binCat.binCat(i) > 2
            binCatCounter = binCatCounter + 1;
            for j = 1 : length(model.binCat.catVals{binCatCounter})-1
                indCounter = indCounter + 1;
                str = num2str(model.binCat.catVals{binCatCounter}(j+1:end)', [' %.' num2str(precision) 'g,']);
                disp(['z' num2str(indCounter) ' = 1, if x' num2str(i) ' is in {' str(1:end-1) '} else = 0']);
            end
        else
            if model.binCat.binCat(i) == 2
                binCatCounter = binCatCounter + 1;
            end
            indCounter = indCounter + 1;
            disp(['z' num2str(indCounter) ' = x' num2str(i)]);
        end
    end
    zx = 'z';
else
    zx = 'x';
end

if isfield(model.binCat, 'minVals')
    minVals = model.binCat.minVals;
else
    minVals = [];
end

if isfield(model, 'rules')
    if model.trainParams.smoothingK > 0
        disp('The decision rules (smoothed):');
    else
        disp('The decision rules:');
    end
    outputRules(model, zx, showNumCases, precision, dealWithNaN);
else
    if model.trainParams.smoothingK > 0
        disp('The tree (smoothed):');
    else
        disp('The tree:');
    end
    outputTree(model.tree, model.trainParams.modelTree, model.binCat.binCatNew, ...
           minVals, 0, zx, showNumCases, precision, dealWithNaN);
end
printinfo(model);

return

function outputTree(node, modelTree, binCatNew, minVals, offset, zx, showNumCases, precision, dealWithNaN)
p = ['%.' num2str(precision) 'g'];
if node.interior
    if binCatNew(node.splitAttr) % a binary variable (might be synthetic)
        str = [repmat(' ',1,offset) 'if ' zx num2str(node.splitAttr) ' == ' num2str(minVals(node.splitAttr),p)];
    else % a continuous variable
        str = [repmat(' ',1,offset) 'if ' zx num2str(node.splitAttr) ' <= ' num2str(node.splitLocation,p)];
    end
    if dealWithNaN && node.nanLeft
        str = [str ' or ' zx num2str(node.splitAttr) ' == ?'];
    end
    disp(str);
    outputTree(node.left, modelTree, binCatNew, minVals, offset + 1, zx, showNumCases, precision, dealWithNaN);
    disp([repmat(' ',1,offset) 'else']);
    outputTree(node.right, modelTree, binCatNew, minVals, offset + 1, zx, showNumCases, precision, dealWithNaN);
    %disp([repmat(' ',1,offset) 'end']);
else
    if modelTree
        if dealWithNaN
            for k = 1 : length(node.modelAttrIdx)
                disp([repmat(' ',1,offset) 'if ' zx num2str(node.modelAttrIdx(k)) ' == ?, ' zx num2str(node.modelAttrIdx(k)) ' = ' num2str(node.modelAttrAvg(k),p)]);
            end
        end
        % print regression model
        str = [repmat(' ',1,offset) 'y = ' num2str(node.modelCoefs(1),p)];
        for k = 1 : length(node.modelAttrIdx)
            if node.modelCoefs(k+1) >= 0
                str = [str ' +'];
            else
                str = [str ' '];
            end
            str = [str num2str(node.modelCoefs(k+1),p) '*' zx num2str(node.modelAttrIdx(k))];
        end
    else
        str = [repmat(' ',1,offset) 'y = ' num2str(node.value,p)];
    end
    if showNumCases
        str = [str ' (' num2str(node.numCases) ')'];
    end
    disp(str);
end
return

function outputRules(model, zx, showNumCases, precision, dealWithNaN)
p = ['%.' num2str(precision) 'g'];
nRules = length(model.rules);
for i = 1 : nRules
    rules = model.rules{i};
    if isempty(rules)
        str = '';
    else
        str = 'if ';
        for j = 1 : length(rules)
            if (j > 1)
                str = [str 'and '];
            end
            rule = rules{j};
            if dealWithNaN && rule.orNan
                str = [str '('];
            end
            if model.binCat.binCatNew(rule.attr) % a binary variable (might be synthetic)
                if rule.le
                    str = [str zx num2str(rule.attr) ' == ' num2str(model.binCat.minVals(rule.attr),p) ' '];
                else
                    str = [str zx num2str(rule.attr) ' == ' num2str(model.binCat.maxVals(rule.attr),p) ' '];
                end
            else % a continuous variable
                if rule.le
                    str = [str zx num2str(rule.attr) ' <= ' num2str(rule.location,p) ' '];
                else
                    str = [str zx num2str(rule.attr) ' > ' num2str(rule.location,p) ' '];
                end
            end
            if dealWithNaN && rule.orNan
                str = [str 'or ' zx num2str(rule.attr) ' == ?) '];
            end
        end
        str = [str 'then '];
    end
    if model.trainParams.modelTree
        % print regression model
        str = [str 'y = ' num2str(model.outcomesCoefs{i}(1),p)];
        for k = 1 : length(model.outcomesAttrIdx{i})
            if model.outcomesCoefs{i}(k+1) >= 0
                str = [str ' +'];
            else
                str = [str ' '];
            end
            str = [str num2str(model.outcomesCoefs{i}(k+1),p) '*' zx num2str(model.outcomesAttrIdx{i}(k))];
        end
    else
        str = [str 'y = ' num2str(model.outcomes(i),p)];
    end
    if showNumCases
        str = [str ' (' num2str(model.outcomesNumCases(i)) ')'];
    end
    if model.trainParams.modelTree && dealWithNaN && (~isempty(model.outcomesAttrIdx{i}))
        str = [str ' (replace missing values: '];
        for k = 1 : length(model.outcomesAttrIdx{i})
            if k > 1
                str = [str ', '];
            end
            str = [str zx num2str(model.outcomesAttrIdx{i}(k)) '=' num2str(model.outcomesAttrAvg{i}(k),p)];
        end
        str = [str ')'];
    end
    disp(str);
end
return
