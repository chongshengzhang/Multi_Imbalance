% Reference:	
% Name: computeHellingerDistance.m
% 
% Purpose: Compute Hellinger distances at discrete intervals for given features
% 
% Authors: Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% Copyright: (c) 2018 Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% This file is a part of Multi_Imbalance software, a software package for multi-class Imbalance learning. 
% 
% Multi_Imbalance software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
% as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
%
% Multi_Imbalance software is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program. 
% If not, see <http://www.gnu.org/licenses/>.

function [feature,distance,threshold] = computeHellingerDistance(features, labels, numBins)

minVals = min(features);
maxVals = max(features);
binSize = (maxVals - minVals) ./ numBins;
Tplus = sum(labels == 1);
Tminus = sum(labels == 0);
minVals = repmat(minVals,[numBins - 1,1]);
binSizes = repmat(binSize,[numBins - 1,1]) .* repmat([1:numBins-1],[length(binSize),1])';
thresholds = minVals + binSizes;
labels = repmat(labels,[1,size(thresholds)]);
[r,c] = size(features);
features = reshape(features,[r 1 c]);
features = repmat(features,[1,size(thresholds,1),1]);
[r,c] = size(thresholds);
thresholds = reshape(thresholds,[1 r c]);
thresholds = repmat(thresholds,[size(labels,1),1]);
Tlplus = sum(labels & (features <= thresholds));
Tlminus = sum(~labels & (features <= thresholds));
Trplus = sum(labels & (features > thresholds));
Trminus = sum(~labels & (features > thresholds));
distance = (sqrt(Tlplus ./ Tplus) - sqrt(Tlminus ./ Tminus)).^2 + (sqrt(Trplus ./ Tplus) - sqrt(Trminus ./ Tminus)).^2;
[~,r,c] = size(distance);
distance = reshape(distance,[r,c]);
[distance,index] = max(distance);
[distance,feature] = max(distance);
threshold = thresholds(1,index(feature),feature);

end