% Reference:	
% Name: sammeCart.m
% 
% Purpose: AdaBoost.M1 using a Multi-class Exponential loss function) has extended AdaBoost in both the update
%          of samples? weights and the classifier combination strategy.
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

function [trainTime,testTime,predictResults] = sammeCart(traindata,trainlabel,testdata,Max_Iter)
tic;
Learners = {};
weight = ones(1, length(trainlabel)) / length(trainlabel);%step 1
AlphaT = zeros(1, Max_Iter);
labels=unique(trainlabel);

%step 2-8
for t = 1 : Max_Iter
    nb = classregtree(traindata,trainlabel,'weights',weight,'method','classification');
    prec=eval(nb,traindata);
    predicted=cellfun(@str2num, prec);
    
    errorRate=sum(weight(find(predicted~=trainlabel)));%step 4
  
    AlphaT(t)=log((1-errorRate)/(errorRate+eps))+log(length(labels)-1);%step 5
    
    Learners{t} = nb;
%     for i=1:length(trainlabel)%step 6
%         if trainlabel(i)~=predicted(i)         
%             weight(i)=weight(i)* exp( AlphaT(t)) ;
%         end
%     end
    
    weight(:)=weight(:)+weight(:) .* ((trainlabel(:)~=predicted(:)) .* exp( AlphaT(t)));
    
    Z = sum(weight);
    weight = weight / Z;%step 7
    
end

trainTime=toc;
tic;

%step 9
Result = zeros(size(testdata, 1),length(labels));

for i = 1 : length(Learners)
    prec1=eval(Learners{i}, testdata);
    lrn_out=cellfun(@str2num, prec1);
    for j=1:length(labels)
       Result(find(lrn_out==labels(j)),j)= Result(find(lrn_out==labels(j)),j)+AlphaT(i);
    end
end

[max_a,ResultR]=max(Result,[],2);
predictResults=ResultR;
for j=1:length(labels)
    predictResults(find(ResultR==j))= labels(j);
end
testTime=toc;
