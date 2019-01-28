% Reference:	
% Name: imECOC.m
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

function [trainTime,testTime,prelabel] = imECOC(traindata,trainlabel,testdata,type,withw)

tic;

[code,ft,labels] = funClassifier(traindata,trainlabel,type); % steps 1-10 of imECOC algorithm

W = funcW(traindata,trainlabel,code,ft,labels);              % steps 11-12 of imECOC algorithm

if withw==0
    W(1:length(ft))=1;
end

trainTime=toc;

tic;

% The following is for the decoding of the imECOC algorithm on the test data;
% the goal is decoding (using the weight W obtained by the function funcw()), i.e., find the
% prediction result (array) is closest to which codeword, then output the corresponding original class label.

pre = funPre(testdata,code,ft,W);                          % steps 14-15
for i=1:length(pre)
    prelabel(i)=labels(pre(i));
end

prelabel= prelabel';
testTime=toc;
