function results = m5ptest(model, Xtst, Ytst)
% m5ptest
% Tests M5' tree, decision rule set, or ensemble of trees on a test data
% set (Xtst, Ytst).
%
% Call:
%   results = m5ptest(model, Xtst, Ytst)
%
% Input:
%   model         : M5' model, decision rule set, or a cell array of M5'
%                   models (if ensemble of trees is to be tested).
%   Xtst, Ytst    : Xtst is a matrix with rows corresponding to testing
%                   observations, and columns corresponding to input
%                   variables. Ytst is a column vector of response values.
%                   Missing values in Xtst must be indicated as NaN.
%
% Output:
%   results       : A structure of different error measures calculated on
%                   the test data set. The structure has the following
%                   fields (if the model is an ensemble, the fields are
%                   column vectors with one (cumulative) value for each
%                   ensemble size, the very last value being error for a
%                   full ensemble):
%     MAE         : Mean Absolute Error.
%     MSE         : Mean Squared Error.
%     RMSE        : Root Mean Squared Error.
%     RRMSE       : Relative Root Mean Squared Error.
%     R2          : Coefficient of Determination.

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

% Last update: April 7, 2016

if nargin < 3
    error('Not enough input arguments.');
end
if isempty(Xtst) || isempty(Ytst)
    error('Data is empty.');
end
if iscell(Xtst) || iscell(Ytst)
    error('Xtst and Ytst should not be cell arrays.');
end
if size(Ytst,2) ~= 1
    error('Ytst should have one column.');
end
if any(any(isnan(Ytst)))
    error('Cannot handle NaNs in Ytst.');
end
n = size(Xtst, 1);
if (n ~= size(Ytst, 1))
    error('The number of rows in Xtst and Ytst should be equal.');
end
numModels = length(model);
if (numModels == 1)
    residuals = m5ppredict(model, Xtst) - Ytst;
else
    YtstRep = repmat(Ytst, 1, numModels);
    residuals = m5ppredict(model, Xtst) - YtstRep;
end
results.MAE = mean(abs(residuals), 1);
results.MSE = mean(residuals .^ 2, 1);
results.RMSE = sqrt(results.MSE);
if n > 1
    variance = var(Ytst, 1);
    results.RRMSE = results.RMSE ./ sqrt(variance);
    results.R2 = 1 - results.MSE ./ variance;
else
    results.RRMSE = Inf(1,numModels);
    results.R2 = -Inf(1,numModels);
end
if (numModels > 1)
    % transpose so that rows correspond to increasing number of trees
    results.MAE = results.MAE';
    results.MSE = results.MSE';
    results.RMSE = results.RMSE';
    results.RRMSE = results.RRMSE';
    results.R2 = results.R2';
end
return
