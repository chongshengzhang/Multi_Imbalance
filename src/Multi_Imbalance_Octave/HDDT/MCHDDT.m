% Reference:	
% Name: MCHDDT.m
% 
% Purpose: Detailed explanations of the HDDT and Multi-class HDDT algorithms are given in the HDDTMC() function.
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
function [time1,time2,predictions]=MCHDDT(traindata,trainlabel,testdata,testlabel)

tic;
trainingLabels = trainlabel ;
trainingFeatures = traindata;
testLabels = testlabel;
testFeatures = testdata;

model = fitHellingerTreeMC(trainingFeatures,trainingLabels);%

time1=toc;
tic;
predictions = predictHellingerTree(model,testFeatures);
time2=toc;
