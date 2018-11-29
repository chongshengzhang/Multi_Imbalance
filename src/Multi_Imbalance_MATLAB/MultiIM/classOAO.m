% Reference:	
% Name: classAO.m
% 
% Purpose: a classification algorithm (originally designed) for binary imbalanced data.
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

function [time1,time2,pre0] = classOAO(train,testdata)
tic;
labels = unique (train(:,end));
numberc=length(labels);
flagc=1;
for i=1:numberc-1
    for j=i+1:numberc
        idi=(train(:,end)==labels(i));
        idj=(train(:,end)==labels(j));
        Dij=[train(idi,:);train(idj,:)];
        pre{flagc} = multiIMCart(Dij(:,1:end-1),Dij(:,end),testdata);
        flagc=flagc+1;
    end
end
time1=toc;

tic;
numbertest=size(testdata,1);
numberC=length(pre);
allpre=zeros(numbertest,numberC);
for t=1:length(pre)
    allpre(:,t)=pre{t};
end

pre0=mode(allpre,2);
time2=toc;
