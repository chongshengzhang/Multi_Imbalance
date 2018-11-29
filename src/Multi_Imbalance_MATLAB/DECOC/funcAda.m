% Reference:	
% Name: funcAda.m
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

function [predicted] = funcAda(train,train_label,test,test_label)

feas=zeros(1,1);
feas=num2cell(feas);
feas{1,1}='class';
for i=1:size(train,2)
    str=['attr',num2str(i)];
    feas=[feas,str];
end
featureNames=feas(1,2:size(feas,2));
featureNames=[featureNames,'class'];

classindex = size(train,2)+1;

traindata=[num2cell(train),num2cell(train_label)];
testdata=[num2cell(test),num2cell(test_label)];

for i=1:size(traindata,1)
    traindata{i,size(traindata,2)}=['class',num2str(traindata{i,size(traindata,2)})];
end

for i=1:size(testdata,1)
    testdata{i,size(testdata,2)}=['class',num2str(testdata{i,size(testdata,2)})];
end

tr = matlab2weka('train',featureNames,traindata,classindex);
ts = matlab2weka('test',featureNames,testdata);

% train

nb = trainWekaClassifier(tr,'meta.AdaBoostM1');


% test

[predicted, classProbs] = wekaClassify(ts,nb);


