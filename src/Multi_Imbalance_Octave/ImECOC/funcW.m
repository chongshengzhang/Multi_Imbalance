% Reference:	
% Name: funcW.m
% 
% Purpose: Learning imbalanced multi-class data with optimal dichotomy weights. IEEE 13th International Conference on Data Mining
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
function W = funcW(traindata,trainlabel,code,ft,labels)

numbertest=size(traindata,1);

W(1:length(ft))=sqrt(1/length(ft));
for t=1:length(ft)
    prec=treeval(ft{t},traindata);
    fX(:,t) = prec;
end

for i=1:length(labels)
    ny(i)=length(find(trainlabel==labels(i)));
end

for i=1:length(labels)
    gama(i)=max(ny)/ny(i);
end

for i=1:numbertest
    ftx=fX(i,:);
    indx=find(labels==trainlabel(i));
    yi=code(indx,:);
    for t=1:length(ftx)
        if ftx(t)~=yi(t)
            btyt=(1-ftx(t)*code(indx,t))/2;    % step 11
            W(t)=W(t)+gama(indx)*btyt;         % step 12
        end
    end
end

W=sqrt(W/sum(W));
