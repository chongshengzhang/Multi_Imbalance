% Reference:	
% Name: multiImCart.m
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
function pre = multiIMCart(traindata,trainlabel,testdata)
% find minority class
table = tabulate(trainlabel,2);
idxt=find(table(:,2)>0);
table=table(idxt,:);

if table(1,2)<table(2,2)
    idxp = (trainlabel()==table(1,1));
    R=ceil(table(2,2)/table(1,2));
else
     idxp = (trainlabel()==table(2,1));
    R=ceil(table(1,2)/table(2,2));
end

P = [traindata(idxp,:),trainlabel(idxp,:)];    % P is the samples of minority class
N = [traindata(~idxp,:),trainlabel(~idxp,:)];  % N is the samples of majority class
numberP=length(trainlabel(idxp,:));

% data{i} is constructed from the original imbalanced dataset to include
% all the samples from the minority class and an equal number
% of samples selected randomly from the majority class.

for i=1:R-1
    data{i}=[P;N(numberP*(i-1)+1:numberP*i,:)];%
end

data{R}=[P;N(numberP*(i-1)+1:end,:)];
for i=1:R
    train=data{R};
    test=[];
    for j=1:R
        if i~=j
            test=[test;data{R}];
        end
    end

    % A PRM model is then learned from each subset.
    ft{i} = treefit(train(:,1:end-1),train(:,end),'method','classification');
    if R==1
        prec=treeval(ft{i},train(:,1:end-1));
        prec = prec;
        ACc=train(:,end)-prec;
        ANc=find(ACc==0);
        acc(i)=size(ANc,1)/length(prec);
    else
        prec=treeval(ft{i},test(:,1:end-1));
        prec = prec;
        ACc=test(:,end)-prec;
        ANc=find(ACc==0);
        acc(i)=size(ANc,1)/length(prec);
    end
    
end

% Once the learning phase is complete, the PRM models are combined
% using the weighting voting strategy, where each model may
% have a different weight for classifying new instances.
weight=acc/sum(acc);
results=zeros(size(testdata,1),2);    
for i=1:R
    prec=treeval(ft{i},testdata);
    prec = prec;
    for j=1:length(prec)
        if prec(j)==table(1,1)
            results(j,1)=results(j,1)+weight(i);
        else 
            results(j,2)=results(j,2)+weight(i);
        end
    end
end
for j=1:size(testdata,1)
    if results(j,1)>results(j,2)
        pre(j)=table(1,1);
        
    else
        pre(j)=table(2,1);
    end
end
