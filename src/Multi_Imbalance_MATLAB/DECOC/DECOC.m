% Reference:	
% Name: DECOC.m
% 
% Purpose: DECOC uses ECOC to tranform the multi-class data into multiple binary data, then finds the best classifier 
%          for each specific binaried data, which will be kept by ft.
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

function [trainTime,testTime,prelabel] = DECOC(traindata,trainlabel,testdata,type,withw)

tic;

[code,ft,labels,D] = funClassifierDECOC(traindata,trainlabel,type);

W = funcwEDOVO(traindata,trainlabel,code,ft,labels,D);

if withw==0
    W(1:length(ft))=1;
end

trainTime=toc;
tic;

pre = funcPreTestEDOVO(testdata,code,ft,W,D);

for i=1:length(pre)
    prelabel(i)=labels(pre(i));
end

prelabel= prelabel';
testTime=toc;
