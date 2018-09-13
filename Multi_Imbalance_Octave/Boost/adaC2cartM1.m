% Reference:
% Sun, Y., Kamel, M. S. & Wang, Y. (2006). Boosting for learning multiple classes with imbalanced class
% distribution. Proceedings of the 6th International Conference on Data Mining, 2006 (PP. 592-602).
%
% AdaC2.M1 (the above paper) derives the best cost setting through the genetic algorithm (GA) method, then
% takes this cost setting into consideration in the subsequent boosting. Genetic algorithm proposed by
% Holland is based on natural selection and genetics of random search technology. GA can achieve
% excellent performance in finding the best parameters.
%
% Since GA is very time consuming, in [15], the authors propose AdaBoost.NC which deprecates the GA
% algorithm, but emphasizes ensemble diversity during training, and exploits its good generalization
% performance to facilitate class imbalance learning [36].
%
% The main steps of AdaC2.M1 are as follows:
% 1: Initialize the weight Vector with uniform distribution
% 2: for t=1 to Max_Iter do
% 3:    Fit a classifier nb to the training data using weights 
% 4:    Compute weighted error
% 5:    Compute AlphaT=0.5*log((rightRate+eps)/(errorRate+eps))
% 6:    Update weights 
% 7:    Re-normalize weight
% 8: end for
% 9: Output Final Classifier
%

function [time1,time2,ResultR0] = adaC2cartM1(traindata,trainlabel,testdata,Max_Iter,C)% C is the optimum cost setup of each class
tic;
Learners = {};
weight = ones(1, length(trainlabel)) / length(trainlabel);%step 1
AlphaT = zeros(1, Max_Iter);
for t = 1 : Max_Iter%step 2-8
    
    nb = treefit(traindata,trainlabel,'weights',weight,'method','classification');%step 3
    prec = treeval(nb,traindata);
    predicted = prec;
    
    Cweight=weight .* C;
    errorRate=sum(Cweight(find(predicted~=trainlabel)));%step 4
    rightRate=sum(Cweight(find(predicted==trainlabel)));
    AlphaT(t)=0.5*log((rightRate+eps)/(errorRate+eps));%step 5
    
    Learners{t} = nb;
    for i=1:length(trainlabel)%step 6
        if trainlabel(i)==predicted(i)
            weight(i)=C(i)*weight(i)* exp(- 1 * ( AlphaT(t)));
        else
            weight(i)=C(i)*weight(i)* exp( AlphaT(t)) ;
        end
    end
    
    Z = sum(weight);
    weight = weight / Z;%step 7
    
end
time1=toc;
tic;
labels=unique(trainlabel);
%step 9
Result = zeros(size(testdata, 1),length(labels));

for i = 1 : length(Learners)    
%     lrn_out =  treeval(Learners{i}, testdata);
    prec1=treeval(Learners{i}, testdata);
    lrn_out = prec1;
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
