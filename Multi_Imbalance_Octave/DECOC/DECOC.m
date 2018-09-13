%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(1) Reference (we design and implement the following algorithm):
%   Jingjun Bi, Chongsheng Zhang*. (2018). An Empirical Comparison on State-of-the-art Multi-class Imbalance
%   Learning Algorithms and A New Diversified Ensemble Learning Scheme.
%   Knowledge-based Systems, 2018, Vol. XXX, pp. XXX.
%
%(2) Using funclassifierDECOC, DECOC uses ECOC to tranform the multi-class data into multiple binary data,
%    then finds the best classifier for each specific binaried data, which will be kept by ft. see line 23.
%
%(3) Using funcwEDOVO, it builds the best classifier for each binarized data (by ECOC) and the predictions are in allpre.
%    Notice that, in this function, it has duplicates with funclassifierDECOC, in building the best classification
%    model for each specific binarized data.  In specific, funcwEDOVO calls funcPreEDOVO,
%    which (retrains) rebuilds the model that funclassifierDECOC has built previously. This should be fixed.
%    This is for the traning data. see line 25.
%
%(4) With funcpretestEDOVO  to make the predictions on the test data, using the imECOC algorithm. see line 34
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [time1,time2,prelabel] = DECOC(traindata,trainlabel,testdata,type,withw)

tic;

[code,ft,labels,D] = funclassifierDECOC(traindata,trainlabel,type);

W = funcwEDOVO(traindata,trainlabel,code,ft,labels,D);

if withw==0
    W(1:length(ft))=1;
end

time1=toc;
tic;

pre = funcpretestEDOVO(testdata,code,ft,W,D);

for i=1:length(pre)
    prelabel(i)=labels(pre(i));
end

prelabel= prelabel';
time2=toc;
