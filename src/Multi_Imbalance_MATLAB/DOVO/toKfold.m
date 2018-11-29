% Reference:	
% Name: toKfold.m
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

function data = toKfold(train,kfold)
 target=train(:,end);
 
 feature1=train(:,1:end-1);
 

 cobj = cvpartition(target, 'kfold', kfold);
 
 

for iter = 1:cobj.NumTestSets
  data(iter).train=feature1(cobj.training(iter),:);
  data(iter).trainlabel=target(cobj.training(iter));
  data(iter).test=feature1(cobj.test(iter),:);
  data(iter).testlabel=target(cobj.test(iter)); 
 end

