function [Yq, contrib] = predictsingle(model, Xq, varMap)
% Called from m5ppredict and m5pbuild.
% Predicts response value for a single query point Xq.
% Assumes that Xq is already pre-processed, i.e., categorical variables
% replaced by binary in the same way as m5pbuild does it.

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

if (nargout > 1)
    if (model.trainParams.modelTree || (model.trainParams.smoothingK ~= 0) || (model.trainParams.extractRules ~= 0))
        error('Extracting input variable contributions is available for unsmoothed regression trees only.');
    elseif ~isfield(model.tree, 'value')
        error('Cannot extract input variable contributions - no predictions stored in interior nodes. Set argument keepNodeInfo to true when calling m5pbuild.');
    end
    extractContributions = true;
    contrib = zeros(1, numel(varMap) + 1);
else
    extractContributions = false;
    contrib = [];
end

if model.trainParams.extractRules == 0
    predictSingleDo(model.tree, 0, 0);
else
    nRules = length(model.rules);
    for i = 1 : nRules
        success = true;
        if i < nRules % the last rule should always succeed
            rules = model.rules{i};
            for j = 1 : length(rules)
                rule = rules{j};
                % NaN is treated as the average of the corresponding
                % variable of the training observations reaching the node
                if ~((isnan(Xq(rule.attr)) && rule.orNan) || ...
                    (rule.le && (Xq(rule.attr) <= rule.location)) || ...
                    ((~rule.le) && (Xq(rule.attr) > rule.location)))
                    success = false;
                    break;
                end
            end
        end
        if success
            if model.trainParams.modelTree
                if ~isempty(model.outcomesAttrIdx{i})
                    % Replace NaNs with the average values of the corresponding
                    % variables of the training observations reaching the node
                    A = Xq(model.outcomesAttrIdx{i});
                    where = isnan(A);
                    A(where) = model.outcomesAttrAvg{i}(where);
                    % Calculate prediction
                    Yq = [1 A] * model.outcomesCoefs{i};
                else
                    Yq = model.outcomesCoefs{i};
                end
            else
                Yq = model.outcomes(i);
            end
            break;
        end
    end
end

    function predictSingleDo(node, splitAttr, nodeValuePrev)
    if extractContributions
        if splitAttr > 0
            for varIdx = 1 : numel(varMap)
                if any(varMap{varIdx} == splitAttr)
                    break;
                end
            end
        else
            varIdx = numel(contrib);
        end
        contrib(varIdx) = contrib(varIdx) + (node.value - nodeValuePrev);
    end
    if node.interior
        % NaN is treated as the average of the corresponding
        % variable of the training observations reaching the node
        if (isnan(Xq(node.splitAttr)) && node.nanLeft) || ...
           (Xq(node.splitAttr) <= node.splitLocation)
            if extractContributions
                predictSingleDo(node.left, node.splitAttr, node.value);
            else
                predictSingleDo(node.left);
            end
        else
            if extractContributions
                predictSingleDo(node.right, node.splitAttr, node.value);
            else
                predictSingleDo(node.right);
            end
        end
    else
        if model.trainParams.modelTree
            if ~isempty(node.modelAttrIdx)
                % Replace NaNs with the average values of the corresponding
                % variables of the training observations reaching the node
                A = Xq(node.modelAttrIdx);
                where = isnan(A);
                A(where) = node.modelAttrAvg(where);
                % Calculate prediction
                Yq = [1 A] * node.modelCoefs;
            else
                Yq = node.modelCoefs;
            end
        else
            Yq = node.value;
        end
    end
    end
end
