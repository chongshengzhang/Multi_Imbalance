% Reference:	
% Name: caculateFunc.m
% 
% Purpose: calculate the accuracy of prelabel
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

function [acc,gmean] = calculateFunc(ACTUAL,PREDICTED)

acc=0;

if size(ACTUAL) > 0
   
   label=unique(ACTUAL);
   k=length(label);

   for m=1:k

       idx = (ACTUAL()==label(m));
   
       p(m) = length(ACTUAL(idx));
       
       if p(m)~=0
       
           tp(m) = sum(ACTUAL(idx)==PREDICTED(idx));
		   
           acc(m)=tp(m)/p(m);
       
        end
    end
end

gmean=1;
nonan=0;

for m=1:k
    if ~isnan(acc(m))
        gmean=gmean*acc(m);
		nonan=nonan+1;
    end
end

acc=length(find(ACTUAL==PREDICTED))/length(ACTUAL);
gmean=gmean^(1/nonan);