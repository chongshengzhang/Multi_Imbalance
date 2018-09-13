% Reference:
% Liu, X. Y., Li, Q. Q. & Zhou Z H. (2013). Learning imbalanced multi-class data with optimal dichotomy
% weights. IEEE 13th International Conference on Data Mining (IEEE ICDM), 2013 (PP.  478-487).
% see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf
%
% The imECOC algorithm includes the following techniques:
% (1) in each binary classifier, it simultaneously considers the between-class and the within-class
%     imbalance; see function funclassifier(), line 19;
% (2) in the training/prediction phase, it assigns different weights to different binary classifiers;
%     see function funcw(), line 21;
% (3) in the prediction phase, it decodes it with weighted distance to obtain the optimal weight of the
%     classifier by minimizing the weighted loss.  see function funcpre(), line 35 and lines 36-38.
%

function [time1,time2,prelabel] = imECOC(traindata,trainlabel,testdata,type,withw)

tic;

[code,ft,labels] = funclassifier(traindata,trainlabel,type); % steps 1-10 of imECOC algorithm

W = funcw(traindata,trainlabel,code,ft,labels);              % steps 11-12 of imECOC algorithm

if withw==0
    W(1:length(ft))=1;
end

time1=toc;

tic;

% The following is for the decoding of the imECOC algorithm on the test data;
% the goal is decoding (using the weight W obtained by the function funcw()), i.e., find the
% prediction result (array) is closest to which codeword, then output the corresponding original class label.

pre = funcpre(testdata,code,ft,W);                          % steps 14-15
for i=1:length(pre)
    prelabel(i)=labels(pre(i));
end

prelabel= prelabel';
time2=toc;
