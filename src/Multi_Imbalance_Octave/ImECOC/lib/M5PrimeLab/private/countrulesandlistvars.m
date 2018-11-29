function [nRules, vars] = countrulesandlistvars(model)
% Called from printinfo (m5pbuild) and m5pcv.
% Counts all rules (equal to the number of leaf nodes) in the tree and
% lists all original variables (not the synthetic ones that are made
% automatically).

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

% Last update: November 2, 2015

if model.trainParams.extractRules == 0
    [nRules, usedVars] = countRulesAndListVarsDo(model.tree);
else
    nRules = length(model.rules);
    usedVars = [];
    for i = 1 : nRules
        rules = model.rules{i};
        for j = 1 : length(rules)
            usedVars = union(usedVars, rules{j}.attr);
        end
        if model.trainParams.modelTree
            usedVars = union(usedVars, model.outcomesAttrIdx{i});
        end
    end
end
vars = [];
for v = 1:length(model.binCat.binCat)
    for u = usedVars(:).'
        if any(model.binCat.varMap{v} == u)
            vars = union(vars, v);
            break;
        end
    end
end

    function [nRules, usedVars] = countRulesAndListVarsDo(node)
    % Counts all rules (equal to the number of leaf nodes) in the tree and
    % lists all (synthetic) variables.
    if node.interior
        [nRules, usedVars] = countRulesAndListVarsDo(node.left);
        [nR, uV] = countRulesAndListVarsDo(node.right);
        nRules = nRules + nR;
        usedVars = union(usedVars, uV);
        usedVars = union(usedVars, node.splitAttr);
    else
        nRules = 1;
        if model.trainParams.modelTree
            usedVars = node.modelAttrIdx;
        else
            usedVars = [];
        end
    end
    end
end
