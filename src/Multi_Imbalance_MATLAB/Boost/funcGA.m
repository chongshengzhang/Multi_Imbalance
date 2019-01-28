% Reference:	
% Name: funcGA.m
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

function gmean=funcGA(x)
%testlabel=[1;2;3;1;2;3;1;1;2;3;1;1;1;2;2;3];
%prelabel=[1;2;3;2;2;3;1;3;2;2;1;1;1;2;2;3];
global train;
% global trainlabel;
labels=unique(train(:,end));
numberall=size(train,1);
numbertrain=floor(numberall*0.8);
train(randperm(numberall),:) = train;
traindata=train(1:numbertrain,1:end-1);
trainlabel=train(1:numbertrain,end);
testdata=train(numbertrain+1:end,1:end-1);
testlabel=train(numbertrain+1:end,end);
for i=1:length(trainlabel)
    indexc=find(labels==trainlabel(i));
    weight(i)=x(indexc);
end
ft = classregtree(traindata,trainlabel,'weights',weight,'method','classification');
prec=eval(ft,testdata);
prec=cellfun(@str2num, prec);
[accc,gmeanc] = calculateFunc(testlabel,prec);
gmean=1/gmeanc;