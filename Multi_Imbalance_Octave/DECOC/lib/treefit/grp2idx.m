function [g,gn] = grp2idx(s)
% GRP2IDX  Create index vector from a grouping variable.
% [G,GN]=GRP2IDX(S) creates an index vector G from the grouping
%   variable S.  S can be a numeric vector, a character matrix (each
%   row representing a group name), or a cell array of strings stored
%   as a column vector.  The result G is a vector taking integer
%   values from 1 up to the number of unique entries in S.  GN is a
%   cell array of names, so that GN(G) reproduces S (aside from any
%   differences in type).

%   Copyright 1993-2004 The MathWorks, Inc. 
%   $Revision: 1.4.4.2 $  $Date: 2004/03/02 21:49:11 $

if (ischar(s))
   s = cellstr(s);
end
if (size(s, 1) == 1)
   s = s';
end

[gn,i,g] = uniquep(s);           % b=unique group names

ii = find(strcmp(gn, ''));
if (length(ii) == 0)
   ii = find(strcmp(gn, 'NaN'));
end

if (length(ii) > 0)
   nangrp = ii(1);        % this group should really be NaN
   gn(nangrp,:) = [];     % remove it from the names array
   g(g==nangrp) = NaN;    % set NaN into the group number array
   g = g - (g > nangrp);  % re-number remaining groups
end



% -----------------------------------------------------------
function [b,i,j] = uniquep(s)
% Same as UNIQUE but orders result:
%    if iscell(s), preserve original order
%    otherwise use numeric order

[b,i,j] = unique(s);     % b=unique group names

nb = size(b,1);
i = zeros(nb,1);
if (iscell(s))
   % Restore in original order
   for k=1:size(b,1)
      ii = find(strcmp(s, b(k)));
      i(k) = ii(1);  % find first instance of each element of b
   end
   isort = i;        % sort based on this order
else
   % If b is a vector, put in numeric order
   for k=1:size(b,1)
      ii = find(s == b(k));
      if (length(ii) > 0)
         i(k) = ii(1); % make sure this is the first instance
      end
   end

   % Fix up bad treatment of NaN
   if (any(isnan(b)))  % remove multiple NaNs; put one at the end
      nans = isnan(b);
      b = [b(~nans); NaN];
      x = find(isnan(s));
      i = [i(~nans); x(1)];
      j(isnan(s)) = length(b);
   end
   
   isort = b;          % sort based on numeric values
   if any(isnan(isort))
      isort(isnan(isort)) = max(isort) + 1;
   end
end

[is, f] = sort(isort); % sort according to the right criterion
b = b(f,:);

[fs, ff] = sort(f);    % rearrange j also
j = ff(j);

if (~iscell(b))        % make sure b is a cell array of strings
   b = cellstr(strjust(num2str(b), 'left'));
end