function trainParams = m5pparams2(varargin)
% m5pparams2
% Creates configuration for building M5' trees or decision rules. The
% output structure is for further use with m5pbuild and m5pcv functions.
% This function is an alternative to function m5pparams for supplying
% parameters as name/value pairs.
%
% Call:
%   trainParams = m5pparams2(varargin)
%
% Input:
%   varargin      : Name/value pairs for the parameters. For the list of
%                   the names, see description of function m5pparams.
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

% Last update: July 28, 2016

nArgs = length(varargin);
if round(nArgs / 2) ~= nArgs / 2
    error('varargin should contain name/value pairs.');
end

trainParams = m5pparams();
paramNames = fieldnames(trainParams);
paramNamesLower = lower(paramNames);

for pair = reshape(varargin, 2, [])
    name = lower(pair{1});
    found = strcmp(name, paramNamesLower);
    if any(found)
        trainParams.(paramNames{found}) = pair{2};
    else
        error('%s is not a recognized parameter name.', name);
    end
end

return
