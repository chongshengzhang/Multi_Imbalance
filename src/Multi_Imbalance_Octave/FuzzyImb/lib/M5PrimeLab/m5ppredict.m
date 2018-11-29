function [Yq, contrib] = m5ppredict(model, Xq)
% m5ppredict
% Predicts response values for the given query points Xq using M5' tree,
% decision rule set, or ensemble of trees. For unsmoothed regression trees
% (whether in ensembles or as individual trees), can also provide a matrix
% of input variable contributions to the response value for each row of Xq.
%
% Call:
%   [Yq, contrib] = m5ppredict(model, Xq)
%
% Input:
%   model         : M5' model, decision rule set, or a cell array of M5'
%                   models, if ensemble of trees is to be used.
%   Xq            : A matrix of query data points. Missing values in Xq
%                   must be indicated as NaN.
%
% Output:
%   Yq            : A column vector of predicted response values. If model
%                   is an ensemble, Yq is a matrix whose rows correspond to
%                   Xq rows (i.e., observations) and columns correspond to
%                   each ensemble size (i.e., the increasing number of
%                   trees), the values in the very last column being the
%                   values for a full ensemble.
%   contrib       : Available only for unsmoothed regression trees (whether
%                   in ensembles or as single trees) and only if m5pbuild
%                   was called with keepNodeInfo = true.
%                   A matrix of contributions of each input variable to the
%                   response for each Xq row in terms of response
%                   value changes along the prediction path of a tree so
%                   that Yq = training_set_mean + x1_contribution +
%                   x2_contribution + ... + xn_contribution. contrib has
%                   the same number of columns as Xq plus one, the last
%                   column being the training set response mean for single
%                   trees or in-bag response mean for ensembles. The sum of
%                   columns of contrib is equal to Yq (or to the last
%                   column of Yq for ensembles).
%                   This allows interpreting single trees and whole
%                   ensembles as well as explaining their predictions. The
%                   implemented method is sometimes also called "feature
%                   contribution method" (Saabas, 2014/2015 (see this for
%                   the simplest explanation); Welling et al., 2016;
%                   Kuz’min et al., 2011; Palczewska et al., 2013). See
%                   also examples of usage in Section 3 of user's manual.
%                   Note that this function does not decompose
%                   contributions according to the Forest Floor methodology
%                   (Welling et al., 2016) as contrib is computed using the
%                   given Xq, instead of the out-of-bag data. For
%                   decomposition in accordance with Forest Floor, see
%                   output argument ensembleResults.OOBContrib of function
%                   m5pbuild.
%
% Remarks:
% 1. If the data contains categorical variables with more than two
%    categories, they are transformed into synthetic binary variables in
%    exactly the same way as m5pbuild does it.
% 2. Any previously unseen values of binary or categorical variables are
%    treated as NaN.

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

if nargin < 2
    error('Not enough input arguments.');
end
if iscell(Xq)
    error('Xq should not be cell array.');
end
[nq, m] = size(Xq);

numModels = length(model);
if (numModels > 1)
    models = model;
    model = models{1};
else
    if iscell(model)
        model = model{1};
    end
end
if length(model.binCat.varMap) ~= size(Xq,2)
    error('The number of columns in Xq is different from the number when the model was built.');
end

% Transform all the categorical variables to binary ones (exactly the same way as with training data)
if any(model.binCat.binCat >= 2)
    binCatCounter = 0;
    synthCounter = 1;
    Xnew = NaN(nq, model.binCat.varMap{end}(end));
    for i = 1 : m
        if model.binCat.binCat(i) > 2
            binCatCounter = binCatCounter + 1;
            %{
            % Warn if a value was not seen when the tree was built
            XX = Xq(:,i);
            XX = unique(XX(~isnan(XX))); % no NaNs
            diff = setdiff(XX, model.binCat.catVals{binCatCounter});
            if ~isempty(diff)
                fprintf('Warning: Categorical variable x%d has one or more previously unseen values: %s. Treating as NaN.', i, mat2str(diff(:)'));
            end
            %}
            len = length(model.binCat.catVals{binCatCounter});
            for j = 1 : len
                where = Xq(:,i) == model.binCat.catVals{binCatCounter}(j);
                Xnew(where,synthCounter:(synthCounter-1 + j-1)) = 1;
                Xnew(where,(synthCounter-1 + j):(synthCounter-1 + len-1)) = 0;
            end
            synthCounter = synthCounter + len-1;
        elseif model.binCat.binCat(i) == 2
            binCatCounter = binCatCounter + 1;
            %{
            % Warn if a value was not seen when the tree was built
            XX = Xq(:,i);
            XX = unique(XX(~isnan(XX))); % no NaNs
            diff = setdiff(XX, model.binCat.catVals{binCatCounter});
            if ~isempty(diff)
                fprintf('Warning: Binary variable x%d has one or more previously unseen values: %s. Treating as NaN.', i, mat2str(diff(:)'));
            end
            %}
            val = Xq(:,i);
            catVals = model.binCat.catVals{binCatCounter};
            where = (val == catVals(1)) | (val == catVals(2));
            Xnew(where,synthCounter) = val(where,1);
            synthCounter = synthCounter + 1;
        else
            Xnew(:,synthCounter) = Xq(:,i);
            synthCounter = synthCounter + 1;
        end
    end
    Xq = Xnew;
end

Yq = zeros(nq,numModels);
if (nargout > 1)
    contrib = zeros(nq, m + 1);
else
    contrib = [];
end

if numModels == 1
    if (nargout > 1)
        for i = 1 : nq
            [Yq(i), contrib(i,:)] = predictsingle(model, Xq(i,:), model.binCat.varMap);
        end
    else
        for i = 1 : nq
            Yq(i) = predictsingle(model, Xq(i,:));
        end
    end
else
    % rows correspond to observations, columns correspond to increasing number of trees
    if (nargout > 1)
        for j = 1 : numModels
            for i = 1 : nq
                [Yq2, contrib2] = predictsingle(models{j}, Xq(i,:), model.binCat.varMap);
                Yq(i,j:numModels) = Yq(i,j:numModels) + Yq2;
                contrib(i,:) = contrib(i,:) + contrib2;
            end
            Yq(:,j) = Yq(:,j) / j;
        end
        contrib = contrib / numModels;
    else
        for j = 1 : numModels
            for i = 1 : nq
                Yq(i,j:numModels) = Yq(i,j:numModels) + predictsingle(models{j}, Xq(i,:));
            end
            Yq(:,j) = Yq(:,j) / j;
        end
    end
end
return
