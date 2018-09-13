% Reference：
% Hoens, T. R., Qian, Q., Chawla, N. V., et al. (2012). Building decision trees for the multi-class imbalance
% problem. Advances in Knowledge Discovery and Data Mining. Springer Berlin Heidelberg, 2012 (PP. 122-134).
%
% Detailed explanations of the HDDT and Multi-class HDDT algorithms are given in the HDDTMC() function.

function [time1,time2,predictions]=MCHDDT(traindata,trainlabel,testdata,testlabel)

tic;
trainingLabels = trainlabel ;
trainingFeatures = traindata;
testLabels = testlabel;
testFeatures = testdata;

model = fit_Hellinger_treeMC(trainingFeatures,trainingLabels);%

time1=toc;
tic;
predictions = predict_Hellinger_tree(model,testFeatures);
time2=toc;
