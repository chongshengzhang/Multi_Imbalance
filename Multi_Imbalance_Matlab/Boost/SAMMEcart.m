% Reference:
% Zhu, J., Zou, H., Rosset, S., et al. (2006). Multi-class AdaBoost. Statistics & Its Interface,
% 2006, 2(3), 349-360.
%
% AdaBoost (Adaptive Boosting) is a binary classification algorithm proposed by Freund and Schapire that
% integrates multiple weak classifiers to build a stronger classifier. AdaBoost only supports binary data in
% the beginning, but it was later extended to multi-class scenarios. AdaBoost.M1 and SAMME (Stagewise
% Additive Modeling using a Multi-class Exponential loss function) have extended AdaBoost in both the update
% of samples’ weights and the classifier combination strategy. The main difference between them is the method
% for updating the weights of the samples.
%
% The main procedure (steps) of the SAMME algorithm:
% Step 1: Initialize the weight Vector with uniform distribution
% Step 2: for t=1 to Max_Iter do
% Step 3:    Fit a classifier nb to the training data using weights
% Step 4:    Compute weighted error: errorRate=sum(weight(find(predicted~=trainlabel)));
% Step 5:    Compute AlphaT=log((1-errorRate)/(errorRate+eps))+log(length(labels)-1)
% Step 6:    Update weights weight(i)=weight(i)* exp( AlphaT(t));(trainlabel(i)~=predicted(i))
% Step 7:    Re-normalize weight
% Step 8: end for
% Step 9: Output Final Classifier
%

function [time1,time2,ResultR0] = SAMMEcart(traindata,trainlabel,testdata,Max_Iter)
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
    for i=1:length(trainlabel)%step 6
        if trainlabel(i)~=predicted(i)         
            weight(i)=weight(i)* exp( AlphaT(t)) ;
        end
    end
    
    Z = sum(weight);
    weight = weight / Z;%step 7
    
end

time1=toc;
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
ResultR0=ResultR;
for j=1:length(labels)
    ResultR0(find(ResultR==j))= labels(j);
end
time2=toc;
