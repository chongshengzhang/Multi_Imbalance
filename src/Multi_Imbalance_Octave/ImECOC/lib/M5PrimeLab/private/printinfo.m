function printinfo(model)
% Called from m5pbuild and m5pprint.
% Prints info about the tree.

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

[nRules, vars] = countrulesandlistvars(model);
fprintf('Number of rules: %d\n', nRules);
if ~isempty(vars)
    list = '';
    for i = 1:length(vars)
        list = [list 'x' int2str(vars(i))];
        if i < length(vars)
            list = [list ', '];
        end
    end
    fprintf('Number of original input variables used: %d (%s)\n', length(vars), list);
else
    fprintf('Number of original input variables used: 0\n');
end
return
