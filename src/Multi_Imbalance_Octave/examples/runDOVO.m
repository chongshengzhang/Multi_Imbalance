% function runDOVO.m
% javaaddpath('weka.jar');
p = genpath(pwd);
addpath(p, '-begin');

load('data\Wine_data_set_indx_fixed.mat'); 
trainData=data(1).train;
trainLabel=data(1).trainlabel;
testData=data(1).test;
testLabel = data(1).testlabel;
 
% the final predicted results for testData will be kept in predictedResults
% the meanings of the rest parameters can be found in the corresponding API reference
[trainTime, testTime, predictedResults,C] = DOVO([trainData,trainLabel],testData,testLabel,5);