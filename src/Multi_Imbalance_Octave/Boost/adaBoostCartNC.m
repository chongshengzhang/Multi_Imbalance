% Reference:	
% Name: adaBoostCartNC.m
% 
% Purpose: adaBoostCartNC using a Multi-class Exponential loss function) have extended AdaBoost in both the update
%          of samples weights and the classifier combination strategy.
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
function [time1,time2,ResultR0] = adaBoostCartNC(traindata,trainlabel,testdata,Max_Iter,lama)

tic;
Learners = {};
weight = ones(1, length(trainlabel)) / length(trainlabel);   % Initialize data weight
AlphaT = zeros(1, Max_Iter);
pt=ones(1,length(trainlabel));% penalty term
amb=zeros(1,length(trainlabel));

% For training epoch t=1
t=1;

% step 1: Train weak classifier nb using distribution weight
nb = treefit(traindata,trainlabel,'weights',weight,'method','classification');
prec=treeval(nb,traindata);

% step 2: get weak classifier
predicted = prec;

errorRate=sum(weight(find(predicted~=trainlabel)));
rightRate=sum(weight(find(predicted==trainlabel)));

% step 4: calculate AlphaT by error and penalty
AlphaT(t)=0.5*log((rightRate+eps)/(errorRate+eps));

Learners{t} = nb;
% step 5: update data weights and obtain new weights by error and penalty
for i=1:length(trainlabel)
    if trainlabel(i)==predicted(i)
        weight(i)=weight(i)* exp(- 1 * ( AlphaT(t)));
    else
        weight(i)=weight(i)* exp( AlphaT(t)*0) ;
    end
end

Z = sum(weight);
weight = weight / Z;

% Output the final ensemble
labels=unique(trainlabel);
trainHt = zeros(size(traindata, 1),length(labels));
for j=1:length(labels)
    trainHt(find(predicted==labels(j)),j)= trainHt(find(predicted==labels(j)),j)+AlphaT(t);
end
[max_a,maxindex]=max(trainHt,[],2);
ResultHt=maxindex;
for j=1:length(labels)
    ResultHt(find(maxindex==j))= labels(j);
end

% For training epoch t = 2 : Max_Iter
for t = 2 : Max_Iter
    
    % 1): Train weak classifier nb using distribution weight
    nb = treefit(traindata,trainlabel,'weights',weight,'method','classification');
    prec=treeval(nb,traindata);

    % 2): get weak classifier
    predicted = prec;

    % 3): calculate the penalty value for every example
    for j=1:length(labels)
        trainHt(find(predicted==labels(j)),j)= trainHt(find(predicted==labels(j)),j)+AlphaT(t);
    end

    [max_a,maxindex]=max(trainHt,[],2);
    ResultHt=maxindex;
    for j=1:length(labels)
        ResultHt(find(maxindex==j))= labels(j);
    end
    
    for i=1:length(trainlabel)
        if ResultHt(i)~=predicted(i)
            if trainlabel(i)==predicted(i)
                amb(i)=amb(i)-1;
            else
                amb(i)=amb(i)+1;
            end
        end
    end

    ambt=amb/t;
    pt=1-ambt;
    ptlama=pt .^ lama;
    Pweight=weight .* ptlama;
    errorRate=sum(Pweight(find(predicted~=trainlabel)));
    rightRate=sum(Pweight(find(predicted==trainlabel)));

    % 4): calculate AlphaT by error and penalty
    AlphaT(t)=0.5*log((rightRate+eps)/(errorRate+eps));
    Learners{t} = nb;

    % 5): update data weights and obtain new weights by error and penalty
    for i=1:length(trainlabel)
        if trainlabel(i)==predicted(i)
            weight(i)=Pweight(i)* exp(- 1 * ( AlphaT(t)));
        else
            weight(i)=Pweight(i)* exp( AlphaT(t)*0) ;
        end
    end
    
    Z = sum(weight);
    weight = weight / Z;
    
end
time1=toc;
tic;

Result = zeros(size(testdata, 1),length(labels));
% Output the Final ensemble
for i = 1 : length(Learners)    
    prec1=treeval(Learners{i}, testdata);
    lrn_out = prec1;
    for j=1:length(labels)
       Result(find(lrn_out==labels(j)),j)= Result(find(lrn_out==labels(j)),j)+AlphaT(i);
    end
end

[max_a,ResultR]=max(Result,[],2);
ResultR0=ResultR;
for j=1:length(labels)
    ResultR0(find(ResultR==j))= labels(j);
end
time2=toc;
