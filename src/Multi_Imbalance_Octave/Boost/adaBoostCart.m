% Reference:	
% Name: adaBoostCart.m
% 
% Purpose: AdaBoost using a Multi-class Exponential loss function) have extended AdaBoost in both the update
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
function ResultR = adaBoostCart(traindata,trainlabel,testdata,Max_Iter)

Learners = {};
weight = ones(1, length(trainlabel)) / length(trainlabel);%step 1
AlphaT = zeros(1, Max_Iter);

for t = 1 : Max_Iter%step 2-8
    
    nb = treefit(traindata,trainlabel,'weights',weight,'method','classification');%step 3
    prec=treeval(nb,traindata);
    predicted = prec;
    
    errorRate=sum(weight(find(predicted~=trainlabel)));%step 4
    AlphaT(t)=0.5*log((1-errorRate)/(errorRate+eps));%step 5
    
    Learners{t} = nb;
    for i=1:length(trainlabel)%step 6
        weight(i)=weight(i)* exp(- 1 * ( AlphaT(t) * trainlabel(i)* predicted(i)));
    end
    
    Z = sum(weight);
    weight = weight / Z;%step 7
    
end
%step 9
Result = zeros(size(testdata, 1),1);

for i = 1 : length(Learners)
    
    lrn_out =  treeval(Learners{i}, testdata);
    lrn_out = lrn_out;
    Result = Result + lrn_out * AlphaT(i);
end

ResultR = sign(Result);