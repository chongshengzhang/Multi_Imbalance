function m5pplot(model, varargin)
% m5pplot
% Plots M5' tree. Does not work with ensembles or decision rule sets.
%
% Call:
%   m5pplot(model, varargin)
%
% In the plotted tree, left child of a node corresponds to outcome 'true'
% and right child to outcome 'false'.
% All the input arguments, except the first one, are optional.
%
% Input:
%   model         : M5' model.
%   varargin      : Name/value pairs of parameters:
%     showNumCases : Whether to show the number of training observations
%                   corresponding to each node. Set to 'all' to show it for
%                   all nodes. Set to 'leaves' to show it for leaves only.
%                   Set to 'off' or any other value to turn it off
%                   (default value = 'all').
%     showSD      : Whether to show standard deviation values corresponding
%                   to each node (default value = false). These values can
%                   also be interpreted as Root Mean Squared Error of each
%                   node in its corresponding partition of the training
%                   dataset. Note that the information is available only if
%                   the tree was built using keepNodeInfo = true.
%     precision   : Number of digits used for any numerical values shown
%                   (default value = 5).
%     dealWithNaN : Whether to display how the tree deals with missing
%                   values (NaN, displayed as '?') (default value =
%                   false).
%     layout      : Graph layout algorithm and tree style. Set to 'oblique'
%                   for semi-optimized layout of a tree with edges that
%                   form oblique angles. Set to 'right' for unoptimized
%                   layout of a tree with edges that form right angles. Set
%                   to 'old' for the old version of the graph layout
%                   algorithm (default value = 'oblique').
%     widthMult   : Edge width multiplier (default value = 1).
%     variableWidth : Whether edge width should reflect the number of
%                   training observations (default value = false).
%     colorize    : Whether to colorize nodes and edges according to the
%                   response values (default value = false). Not available
%                   for model trees or if layout = 'old'. Complete
%                   colorization is available only if the regression tree
%                   was built using keepNodeInfo = true.
%     fontSize    : Font size for text (default value = 10).
%
% Remarks:
% 1. For smoothed M5' trees, the smoothing process is already done in
%    m5pbuild, therefore if one wants to see unsmoothed versions (which are
%    usually easier to interpret), the trees should be built with smoothing
%    disabled.
% 2. If the training data has categorical variables with more than two
%    categories, the corresponding synthetic binary variables are shown.
% 3. For unsmoothed regression trees, if they were built using
%    keepNodeInfo = true, the plot will show predicted values at
%    interior nodes as well.

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

modelIsDecisionRules = isfield(model, 'rules');
if modelIsDecisionRules
    error('Decision rule sets cannot be plotted.');
end

showNumCases = 'all';
showSD = false;
precision = 5;
dealWithNaN = false;
widthMult = 1;
variableWidth = false;
fontSize = 10;
colorize = false;
layout = 'oblique';

nArgs = length(varargin);
if round(nArgs / 2) ~= nArgs / 2
    error('varargin should contain name/value pairs.');
end
for pair = reshape(varargin, 2, [])
    switch lower(pair{1})
        case lower('showNumCases'), showNumCases = pair{2};
        case lower('showSD'), showSD = pair{2};
        case lower('precision'), precision = round(pair{2});
        case lower('dealWithNaN'), dealWithNaN = pair{2};
        case lower('widthMult'), widthMult = pair{2};
        case lower('variableWidth'), variableWidth = pair{2};
        case lower('fontSize'), fontSize = pair{2};
        case lower('colorize'), colorize = pair{2};
        case lower('layout'), layout = pair{2};
        otherwise, error('%s is not a recognized parameter name.', name);
    end
end

if model.trainParams.modelTree
    colorize = false;
end
if showSD && ~isfield(model.tree, 'sd')
    showSD = false;
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

if model.trainParams.modelTree
    if model.trainParams.smoothingK > 0
        disp('Models (smoothed):');
    else
        disp('Models:');
    end
end
p = ['%.' num2str(precision) 'g'];
octave = exist('OCTAVE_VERSION', 'builtin');
minY = Inf;
maxY = -Inf;

if strcmpi(layout, 'old')
    
    len = []; % number of nodes in a column
    analyzeChildren(model.tree, 1);
    pos = zeros(max(len), length(len)); % positions of nodes in columns
    for i = 1 : length(len)
        num = len(i);
        step = 1 / (num + 1);
        where = 1:-step:0;
        pos(1:num,i) = where(2:end-1);
    end
    idx = zeros(1, length(len)); % index of current positions in columns
    numModel = 1;
    figure('Color', [1, 1, 1]);
    axis off;
    hold on;
    if colorize
        colorbar;
        caxis([minY maxY]);
    end
    plotChildren(model.tree, 0, 1, model.trainParams.modelTree, model.binCat.binCatNew);
    
else
    
    treeVec = []; % vector that will represent the connectivity of the tree
    tnodes = {}; % list of nodes in the order of treeVec
    ty2 = []; % list of y coords for the plot in the order of treeVec
    analyzeChildrenForTreeVec(model.tree, 1, 0); % populate lists
    tx = treelayout(treeVec); % create list of x coords for the plot in the order of treeVec
    if numel(tx) > 1
        tx = (tx - min(tx)) / (max(tx) - min(tx)); % normalize x coords
        if octave
            tx = 1 - tx;
        end
    else
        tx = 0.5;
    end
    % additional thouch-up of the layout
    if numel(treeVec) > 1
        if ~strcmpi(layout, 'right')
            baseDist = layoutTouchUp(true, []);
            layoutTouchUp(true, baseDist);
            layoutTouchUp(true, baseDist);
            layoutTouchUp(true, baseDist);
            layoutTouchUp(true, baseDist);
        end
        ty2 = (max(ty2-1) - (ty2-1)) / max(ty2-1); % normalize y coords
    else
        ty2 = 0.5;
    end
    % show the tree
    fig = figure('Color', [1, 1, 1]);
    resizeCallbackFcn(fig, true);
    
    if ~octave
        % callback for replotting the tree so that all the text is nicely placed for any plot size
        set(fig, 'ResizeFcn', @(x,y) resizeCallbackFcn(fig, false));
    end
    
end

    function resizeCallbackFcn(fig, printInfo)
        clf(fig);
        axis off;
        hold on;
        if colorize && (minY < maxY)
            axis([0 1.03 0 1]);
            colorbar;
            caxis([minY maxY]);
        end
        plotTreeVec(tx, ty2, treeVec, tnodes, zx, model.trainParams.modelTree, model.binCat.binCatNew, ...
            dealWithNaN, fontSize, widthMult, variableWidth, minVals, showNumCases, showSD, p, printInfo, ...
            strcmpi(layout, 'right'), colorize, minY, maxY, octave);
    end

    function baseDist = layoutTouchUp(withListUpdate, baseDist)
        numLayers = max(ty2);
        nodesXInLayers = cell(numLayers,1); % x coords of all nodes separately for each layer
        nodesXAllLeaves = []; % x coords of all leaves
        for iNode = 1 : numel(tnodes)
            nodesXInLayers{ty2(iNode)} = [nodesXInLayers{ty2(iNode)} tx(iNode)];
            if isempty(baseDist) && ~tnodes{iNode}.interior
                nodesXAllLeaves = [nodesXAllLeaves tx(iNode)];
            end
        end
        if isempty(baseDist)
            % get distance between any two nearest x coords
            % (in the beginning distances between any two x coords of all leaves are equal)
            nodesXAllLeaves = sort(nodesXAllLeaves);
            baseDist = nodesXAllLeaves(2) - nodesXAllLeaves(1);
        end
        for iNode = 1 : numel(tnodes) % loop through all nodes
            layer = ty2(iNode);
            listX = nodesXInLayers{layer};
            idx = find(listX == tx(iNode), 1); % index of x coord for the current node
            if (idx == 1) % the node is first from left
                if listX(idx) <= 0
                    continue; % the node may not move left
                else
                    if tnodes{iNode}.interior
                        distLeft = sqrt(baseDist) * 0.3;
                    else
                        distLeft = sqrt(baseDist) * 0.55;
                    end
                end
            else
                % check, how far left the node can be moved without
                % overlapping with other nodes and going too far from its parent
                if tnodes{iNode}.interior
                    if tnodes{iNode}.bothChildrenLeaves % in this case we may move it more
                        distLeft = min(tx(iNode) - listX(idx-1), sqrt(baseDist) * 0.4);
                    else
                        distLeft = min(tx(iNode) - listX(idx-1), sqrt(baseDist) * 0.3);
                    end
                else
                    distLeft = min(tx(iNode) - listX(idx-1), sqrt(baseDist) * 0.55);
                end
            end
            if (idx == numel(listX)) % the node is first from right
                if listX(idx) >= 1
                    continue; % the node may not move right
                else
                    if tnodes{iNode}.interior
                        distRight = sqrt(baseDist) * 0.3;
                    else
                        distRight = sqrt(baseDist) * 0.55;
                    end
                end
            else
                % check, how far right the node can be moved without
                % overlapping with other nodes and going too far from its parent
                if tnodes{iNode}.interior
                    if tnodes{iNode}.bothChildrenLeaves % in this case we may move it more
                        distRight = min(listX(idx+1) - tx(iNode), sqrt(baseDist) * 0.4);
                    else
                        distRight = min(listX(idx+1) - tx(iNode), sqrt(baseDist) * 0.3);
                    end
                else
                    distRight = min(listX(idx+1) - tx(iNode), sqrt(baseDist) * 0.55);
                end
            end
            
            % the side nodes may not move inwards
            if (idx == 1) && (distRight > distLeft)
                continue;
            end
            if (idx == numel(listX)) && (distLeft > distRight)
                continue;
            end
            
            dist = (-distLeft + distRight) / 2;
            tx(iNode) = tx(iNode) + dist; % the actual movement
            tx(iNode) = max(tx(iNode), 0);
            tx(iNode) = min(tx(iNode), 1);
            if withListUpdate % in some cases the plot looks better if at first this is false. but there is a risk of overlapping etc.
                listX(idx) = tx(iNode);
                nodesXInLayers{layer} = listX; % update the list with the new coords
            end
            
            % for interior nodes, let's try to move their children about the same amount
            if tnodes{iNode}.interior
                % get children indices for the list of nodes
                which = find(treeVec == iNode);
                % get children indices for the list of coords
                listX = nodesXInLayers{layer + 1};
                idx1 = find(listX == tx(which(1)), 1);
                idx2 = find(listX == tx(which(2)), 1);
                % decide on by what amount the children will be moved, so
                % that they don't overlap with other nodes and don't go too far etc.
                if (dist < 0) && (idx1 > 1)
                    dist = max(dist, (listX(idx1-1) - listX(idx1)) * 0.99);
                elseif (dist > 0) && (idx2 < numel(listX))
                    dist = min(dist, (listX(idx2+1) - listX(idx2)) * 0.99);
                end
                if listX(idx1) + dist < 0
                    dist = -listX(idx1);
                elseif listX(idx2) + dist > 1
                    dist = 1 - listX(idx2);
                end
                % move the children
                tx(which) = tx(which) + dist;
                % update the list with the new coords
                listX(idx1) = tx(which(1));
                listX(idx2) = tx(which(2));
                nodesXInLayers{layer + 1} = listX;
            end
            
        end
    end

    function analyzeChildrenForTreeVec(node, depth, prev)
        treeVec = [treeVec prev];
        curr = length(treeVec);
        nodeTmp.interior = node.interior;
        ty2 = [ty2 depth];
        nodeTmp.numCases = node.numCases;
        if isfield(node, 'sd')
            nodeTmp.sd = node.sd;
        end
        if node.interior
            if (~model.trainParams.modelTree) && isfield(node, 'value')
                nodeTmp.value = node.value;
                minY = min(minY, nodeTmp.value);
                maxY = max(maxY, nodeTmp.value);
            end
            nodeTmp.splitAttr = node.splitAttr;
            nodeTmp.splitLocation = node.splitLocation;
            nodeTmp.nanLeft = node.nanLeft;
            nodeTmp.bothChildrenLeaves = (~node.left.interior) && (~node.right.interior);
            tnodes{1,curr} = nodeTmp;
        else
            if ~model.trainParams.modelTree
                nodeTmp.value = node.value;
                minY = min(minY, nodeTmp.value);
                maxY = max(maxY, nodeTmp.value);
            else
                nodeTmp.modelCoefs = node.modelCoefs;
                nodeTmp.modelAttrIdx = node.modelAttrIdx;
                if ~isempty(node.modelAttrIdx)
                    nodeTmp.modelAttrAvg = node.modelAttrAvg;
                end
            end
            tnodes{1,curr} = nodeTmp;
            return;
        end
        analyzeChildrenForTreeVec(node.left, depth + 1, curr);
        analyzeChildrenForTreeVec(node.right, depth + 1, curr);
    end

    function analyzeChildren(node, depth)
        if length(len) >= depth
            len(depth) = len(depth) + 1;
        else
            len(depth) = 1;
        end
        if (~model.trainParams.modelTree) && isfield(node, 'value')
            minY = min(minY, node.value);
            maxY = max(maxY, node.value);
        end
        if ~node.interior
            return;
        end
        analyzeChildren(node.left, depth + 1);
        analyzeChildren(node.right, depth + 1);
    end

    function plotChildren(node, x, depth, modelTree, binCatNew)
        idx(depth) = idx(depth) + 1;
        myY = pos(idx(depth), depth);
        
        if colorize
            cmap = colormap;
        else
            cmap = [];
        end
        
        if node.interior
            newX = x - 10;
            if ~isempty(node.left)
                newY = pos(idx(depth + 1) + 1, depth + 1);
                if variableWidth
                    plot(-[myY;newY], [x;newX], '-', 'Color', getColor(node.left, cmap, colorize, minY, maxY), ...
                        'LineWidth', (15 * node.left.numCases / model.tree.numCases + 0.5) * widthMult);
                else
                    plot(-[myY;newY], [x;newX], '-', 'Color', getColor(node.left, cmap, colorize, minY, maxY), ...
                        'LineWidth', 0.5 * widthMult);
                end
                plotChildren(node.left, newX, depth + 1, modelTree, binCatNew);
            end
            if ~isempty(node.right)
                newY = pos(idx(depth + 1) + 1, depth + 1);
                if variableWidth
                    plot(-[myY;newY], [x;newX], '-', 'Color', getColor(node.right, cmap, colorize, minY, maxY), ...
                        'LineWidth', (15 * node.right.numCases / model.tree.numCases + 0.5) * widthMult);
                else
                    plot(-[myY;newY], [x;newX], '-', 'Color', getColor(node.right, cmap, colorize, minY, maxY), ...
                        'LineWidth', 0.5 * widthMult);
                end
                plotChildren(node.right, newX, depth + 1, modelTree, binCatNew);
            end
        end
        
        if octave
            sizeCoef = 0.5;
        else
            sizeCoef = 1;
        end
        
        if variableWidth
            plot(-myY, x, '.', 'Color', getColor(node, cmap, colorize, minY, maxY), ...
                'MarkerSize', max(15, sqrt(2500 * node.numCases / model.tree.numCases)) * sizeCoef * widthMult);
        else
            plot(-myY, x, '.', 'Color', getColor(node, cmap, colorize, minY, maxY), ...
                'MarkerSize', 15 * sizeCoef * sqrt(widthMult));
        end
        
        % show values
        showValues = (~modelTree) && ((~node.interior) || isfield(node, 'value'));
        if showValues
            text(-myY, x - 1.6, num2str(node.value,p), ...
                'FontSize', fontSize, 'HorizontalAlignment', 'center');
        end
        
        % show numCases
        if ((~node.interior) && strcmpi(showNumCases, 'leaves')) || strcmpi(showNumCases, 'all')
            if showValues || (~node.interior)
                dist = 3.65;
            else
                dist = 1.6;
            end
            text(-myY, x - dist, ['(' num2str(node.numCases) ')'], ...
                'FontSize', fontSize, 'HorizontalAlignment', 'center');
        end
        
        if ~node.interior
            if modelTree
                % print regression model
                str = ['M' num2str(numModel) ' = ' num2str(node.modelCoefs(1),p)];
                for k = 1 : length(node.modelAttrIdx)
                    if node.modelCoefs(k+1) >= 0
                        str = [str ' +'];
                    else
                        str = [str ' '];
                    end
                    str = [str num2str(node.modelCoefs(k+1),p) '*' zx num2str(node.modelAttrIdx(k))];
                end
                if dealWithNaN && (~isempty(node.modelAttrIdx))
                    str = [str ' (replace missing values: '];
                    for k = 1 : length(node.modelAttrIdx)
                        if k > 1
                            str = [str ', '];
                        end
                        str = [str zx num2str(node.modelAttrIdx(k)) '=' num2str(node.modelAttrAvg(k),p)];
                    end
                    str = [str ')'];
                end
                disp(str);
                str = ['M' num2str(numModel)];
                text(-myY, x - 1.65, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
                numModel = numModel + 1;
            end
            if (depth == 1) % if the tree has only one node - the root node
                plot(-[myY;myY], [x+10;x+10], 'w');
                plot(-myY, x, '.', 'Color', getColor(node, cmap, colorize, minY, maxY), ...
                    'MarkerSize', 15 * sizeCoef * sqrt(widthMult));
                plot(-[myY;myY], [x-10;x-10], 'w');
            end
            return;
        end
        
        if binCatNew(node.splitAttr) % a binary variable (might be synthetic)
            str = ([zx num2str(node.splitAttr) '==' num2str(minVals(node.splitAttr),p)]);
            if dealWithNaN && node.nanLeft
                if depth == 1
                    % one line for root, so that the text does not go outside the image
                    str = [str ' or ' zx num2str(node.splitAttr) '==?'];
                else
                    text(-myY, x + 4, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
                    str = ['or ' zx num2str(node.splitAttr) '==?'];
                end
            end
        else % a continuous variable
            str = ([zx num2str(node.splitAttr) '<=']);
            if depth == 1
                % one line for root, so that the text does not go outside the image
                str = [str num2str(node.splitLocation,p)];
                if dealWithNaN && node.nanLeft
                    str = [str ' or ' zx num2str(node.splitAttr) '==?'];
                end
            else
                if dealWithNaN && node.nanLeft
                    str = [str num2str(node.splitLocation,p)];
                    text(-myY, x + 4, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
                    str = ['or ' zx num2str(node.splitAttr) '==?'];
                else
                    text(-myY, x + 4, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
                    str = num2str(node.splitLocation,p);
                end
            end
        end
        text(-myY, x + 2, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
    end
end

function plotTreeVec(tx, ty, treeVec, tnodes, zx, modelTree, binCatNew, dealWithNaN, fontSize, ...
    widthMult, variableWidth, minVals, showNumCases, showSD, p, printInfo, rightAngles, colorize, minY, maxY, octave)
set(gcf, 'Units', 'pixels');
set(gca, 'Units', 'pixels');
ap = get(gca, 'Position');
if numel(treeVec) > 1
    coef = 50 / (ap(4) - ap(2)); % for better text positioning
else
    coef = 1;
end

if octave
    sizeCoef = 0.5;
else
    sizeCoef = 1;
end

if colorize
    cmap = colormap;
else
    cmap = [];
end

maxNumCases = tnodes{1}.numCases;
numModel = 1;
hasValuesInInterior = (~modelTree) && isfield(tnodes{1}, 'value');

% plot edges
for i = 1 : numel(treeVec)
    node = tnodes{i};
    if treeVec(i) > 0
        if variableWidth
            if rightAngles
                plot([tx(i);tx(i);tx(treeVec(i))], [ty(i);ty(treeVec(i));ty(treeVec(i))], '-', ...
                    'Color', getColor(node, cmap, colorize, minY, maxY), ...
                    'LineWidth', (15 * node.numCases / maxNumCases + 0.5) * widthMult);
            else
                plot([tx(i);tx(treeVec(i))], [ty(i);ty(treeVec(i))], '-', ...
                    'Color', getColor(node, cmap, colorize, minY, maxY), ...
                    'LineWidth', (15 * node.numCases / maxNumCases + 0.5) * widthMult);
            end
        else
            if rightAngles
                plot([tx(i);tx(i);tx(treeVec(i))], [ty(i);ty(treeVec(i));ty(treeVec(i))], '-', ...
                    'Color', getColor(node, cmap, colorize, minY, maxY), ...
                    'LineWidth', 0.5 * widthMult);
            else
                plot([tx(i);tx(treeVec(i))], [ty(i);ty(treeVec(i))], '-', ...
                    'Color', getColor(node, cmap, colorize, minY, maxY), ...
                    'LineWidth', 0.5 * widthMult);
            end
        end
    end
end

% plot nodes and part of the text
for i = 1 : numel(treeVec)
    node = tnodes{i};
    
    if variableWidth
        plot(tx(i), ty(i), '.', ...
            'Color', getColor(node, cmap, colorize, minY, maxY), ...
            'MarkerSize', max(15, sqrt(2500 * node.numCases / maxNumCases)) * sizeCoef * widthMult);
    else
        plot(tx(i), ty(i), '.', ...
            'Color', getColor(node, cmap, colorize, minY, maxY), ...
            'MarkerSize', 15 * sizeCoef * sqrt(widthMult));
    end
    
    if ~node.interior
        if modelTree
            % print regression model
            if printInfo
                str = ['M' num2str(numModel) ' = ' num2str(node.modelCoefs(1),p)];
                for k = 1 : length(node.modelAttrIdx)
                    if node.modelCoefs(k+1) >= 0
                        str = [str ' +'];
                    else
                        str = [str ' '];
                    end
                    str = [str num2str(node.modelCoefs(k+1),p) '*' zx num2str(node.modelAttrIdx(k))];
                end
                if dealWithNaN && (~isempty(node.modelAttrIdx))
                    str = [str ' (replace missing values: '];
                    for k = 1 : length(node.modelAttrIdx)
                        if k > 1
                            str = [str ', '];
                        end
                        str = [str zx num2str(node.modelAttrIdx(k)) '=' num2str(node.modelAttrAvg(k),p)];
                    end
                    str = [str ')'];
                end
                disp(str);
            end
            str = ['M' num2str(numModel)];
            text(tx(i), ty(i) - 0.16 * coef * fontSize * 0.1, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
            numModel = numModel + 1;
        end
        if (numel(treeVec) == 1) % if the tree has only one node - the root node
            plot([tx;tx], [ty+0.25;ty+0.25], 'w');
            plot([tx;tx], [ty-0.25;ty-0.25], 'w');
        end
        continue;
    end
    
    if binCatNew(node.splitAttr) % a binary variable (might be synthetic)
        str = ([zx num2str(node.splitAttr) '==' num2str(minVals(node.splitAttr),p)]);
        if dealWithNaN && node.nanLeft
            if i == 1 % root node
                % one line for root, so that the text does not go outside the image
                str = [str ' or ' zx num2str(node.splitAttr) '==?'];
            else
                text(tx(i), ty(i) + 0.4 * coef * fontSize * 0.1, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
                str = ['or ' zx num2str(node.splitAttr) '==?'];
            end
        end
    else % a continuous variable
        str = ([zx num2str(node.splitAttr) '<=']);
        if i == 1 % root node
            % one line for root, so that the text does not go outside the image
            str = [str num2str(node.splitLocation,p)];
            if dealWithNaN && node.nanLeft
                str = [str ' or ' zx num2str(node.splitAttr) '==?'];
            end
        else
            if dealWithNaN && node.nanLeft
                str = [str num2str(node.splitLocation,p)];
                text(tx(i), ty(i) + 0.4 * coef * fontSize * 0.1, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
                str = ['or ' zx num2str(node.splitAttr) '==?'];
            else
                text(tx(i), ty(i) + 0.4 * coef * fontSize * 0.1, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
                str = num2str(node.splitLocation,p);
            end
        end
    end
    text(tx(i), ty(i) + 0.2 * coef * fontSize * 0.1, str, 'FontSize', fontSize, 'HorizontalAlignment', 'center');
end

% show values
if ~modelTree
    for i = 1 : numel(treeVec)
        node = tnodes{i};
        if hasValuesInInterior || (~node.interior)
            text(tx(i), ty(i) - 0.16 * coef * fontSize * 0.1, num2str(node.value,p), ...
                'FontSize', fontSize, 'HorizontalAlignment', 'center');
        end
    end
end

% show numCases and sd
if showSD || strcmp(showNumCases, 'all') || strcmp(showNumCases, 'leaves')
    for i = 1 : numel(treeVec)
        node = tnodes{i};
        if node.interior
            if hasValuesInInterior
                dist = 0.36;
            else
                dist = 0.16;
            end
            if strcmp(showNumCases, 'all')
                text(tx(i), ty(i) - dist * coef * fontSize * 0.1, ['(' num2str(node.numCases) ')'], ...
                    'FontSize', fontSize, 'HorizontalAlignment', 'center');
                dist = dist + 0.2;
            end
            if showSD
                text(tx(i), ty(i) - dist * coef * fontSize * 0.1, ['SD=' num2str(node.sd,'%.3g')], ...
                    'FontSize', fontSize, 'HorizontalAlignment', 'center');
            end
        else
            dist = 0.36;
            if strcmp(showNumCases, 'all') || strcmp(showNumCases, 'leaves')
                text(tx(i), ty(i) - dist * coef * fontSize * 0.1, ['(' num2str(node.numCases) ')'], ...
                    'FontSize', fontSize, 'HorizontalAlignment', 'center');
                dist = dist + 0.2;
            end
            if showSD
                text(tx(i), ty(i) - dist * coef * fontSize * 0.1, ['SD=' num2str(node.sd,'%.3g')], ...
                    'FontSize', fontSize, 'HorizontalAlignment', 'center');
            end
        end
    end
end

end

function c = getColor(node, cmap, colorize, minY, maxY)
if colorize && isfield(node, 'value')
    if minY < maxY
        c = cmap(1 + round((size(cmap,1)-1) * (node.value - minY) / (maxY - minY)),:);
    else
        c = cmap(1,:);
    end
else
    c = [0 0.447 0.741];
end
end
