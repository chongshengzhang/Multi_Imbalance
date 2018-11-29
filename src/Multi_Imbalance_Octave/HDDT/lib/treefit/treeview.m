function  varargout = treeview(T,Action,SI,varargin)
%TREEVIEW TreeView actions interface
%
%  TREEVIEW(T,action,SubItem,varargin)
%
%  Actions and arguments are:
%
%    Clear:  {Tree}
%    Remove: {Tree}
%    Add:    {Tree,ImageList manager, TypeFilter, MaxLvl, [alt parent]}
%    Select: {Tree}
%    LateSelect: {Tree}
%    Refresh: {Tree,ImageList manager, TypeFilter, MaxLvl}
%    Current: {Tree}
%    Currentsub: {Tree}
%    Update: {Tree,ImageList manager}

%  Copyright 2000-2004 The MathWorks, Inc. and Ford Global Technologies, Inc.

%  $Revision: 1.1.6.2 $  $Date: 2004/02/09 08:38:48 $


if nargout==1
   varargout{1} = [];
end

switch lower(Action)
    case 'update'
        i_Update(T,SI,varargin{:});
    otherwise
        % pass on to cgnode
        varargout{1}= treeview(T.cgnode,Action,SI,varargin{:});
end



% Overloaded update that knows about null/non-null subitems
function i_Update(T,SI,h,IL)
nodes = get(h,'nodes');
key = genkey(T,SI);
item = nodes.Item(key);
if isnull(SI)
    % An update is only needed for the setup tradeoff node
    ic = bmp2ind(IL,iconfile(T));
    set(item,'image',ic);
    set(item,'text',name(T));
end
