% Reference:	
% Name: adaC2CartM1.m
% 
% Purpose: AdaBoost.M1 using a Multi-class Exponential loss function) have extended AdaBoost in both the update
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
function [time1,time2,ResultR0] = adaC2CartM1(traindata,trainlabel,testdata,Max_Iter,C)% C is the optimum cost setup of each class
tic;
Learners = {};
weight = ones(1, length(trainlabel)) / length(trainlabel);%step 1
AlphaT = zeros(1, Max_Iter);
for t = 1 : Max_Iter%step 2-8
    
    nb = treefit(traindata,trainlabel,'weights',weight,'method','classification');%step 3
    prec = treeval(nb,traindata);
    predicted = prec;
    
    Cweight=weight .* C;
    errorRate=sum(Cweight(find(predicted~=trainlabel)));%step 4
    rightRate=sum(Cweight(find(predicted==trainlabel)));
    AlphaT(t)=0.5*log((rightRate+eps)/(errorRate+eps));%step 5
    
    Learners{t} = nb;
    for i=1:length(trainlabel)%step 6
        if trainlabel(i)==predicted(i)
            weight(i)=C(i)*weight(i)* exp(- 1 * ( AlphaT(t)));
        else
            weight(i)=C(i)*weight(i)* exp( AlphaT(t)) ;
        end
    end
    
    Z = sum(weight);
    weight = weight / Z;%step 7
    
end
time1=toc;
tic;
labels=unique(trainlabel);
%step 9
Result = zeros(size(testdata, 1),length(labels));

for i = 1 : length(Learners)    
%     lrn_out =  treeval(Learners{i}, testdata);
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
