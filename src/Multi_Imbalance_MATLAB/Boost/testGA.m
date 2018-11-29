% Reference:	
% Name: testGA.m
% 
% Purpose: a mathod for GA test.
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

function C=testGA(traindata,trainlabel)
global train;
train=[traindata,trainlabel];
labels=unique(train(:,end));
ObjectiveFunction = @funcGA;
nvars = length(labels);    % Number of variables
for i=1:nvars
    LB(i) = 1e-5;  
end% Lower bound
UB = ones(1,nvars);  % Upper bound

[x,fval] = ga(ObjectiveFunction,nvars,[],[],[],[],LB,UB,[]);
x0=x/sum(x);
for i=1:length(trainlabel)
    indexc=find(labels==trainlabel(i));
    C(i)=x0(indexc);
end

    