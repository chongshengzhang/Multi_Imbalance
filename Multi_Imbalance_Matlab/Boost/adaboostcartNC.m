% Reference:
% Wang, S., Chen, H. & Yao, X. Negative correlation learning for classification ensembles. Proc. Int. Joint
% Conf. Neural Netw., 2010 (PP. 2893-2900).
%
% AdaC2.M1 (adaC2cartM1) derives the best cost setting through the genetic algorithm (GA) method, then
% takes this cost setting into consideration in the subsequent boosting. Genetic algorithm proposed by
% Holland is based on natural selection and genetics of random search technology. GA can achieve
% excellent performance in finding the best parameters.
%
% Since GA is very time consuming, in the above reference, the authors propose AdaBoost.NC which
% deprecates the GA algorithm, but emphasizes ensemble diversity during training, and exploits its
% good generalization performance to facilitate class imbalance learning.
%

function [time1,time2,ResultR0] = adaboostcartNC(traindata,trainlabel,testdata,Max_Iter,lama)

tic;
Learners = {};
weight = ones(1, length(trainlabel)) / length(trainlabel);   % Initialize data weight
AlphaT = zeros(1, Max_Iter);
pt=ones(1,length(trainlabel));% penalty term
amb=zeros(1,length(trainlabel));

% For training epoch t=1
t=1;

% step 1: Train weak classifier nb using distribution weight
nb = classregtree(traindata,trainlabel,'weights',weight,'method','classification');
prec=eval(nb,traindata);

% step 2: get weak classifier
predicted=cellfun(@str2num, prec);

errorRate=sum(weight(find(predicted~=trainlabel)));
rightRate=sum(weight(find(predicted==trainlabel)));

% step 4: calculate AlphaT by error and penalty
AlphaT(t)=0.5*log((rightRate+eps)/(errorRate+eps));

Learners{t} = nb;
% step 5: update data weights and obtain new weights by error and penalty
for i=1:length(trainlabel)
    if trainlabel(i)==predicted(i)
        weight(i)=weight(i)* exp(- 1 * ( AlphaT(t)));
    else
        weight(i)=weight(i)* exp( AlphaT(t)*0) ;
    end
end

Z = sum(weight);
weight = weight / Z;

% Output the final ensemble
labels=unique(trainlabel);
trainHt = zeros(size(traindata, 1),length(labels));
for j=1:length(labels)
    trainHt(find(predicted==labels(j)),j)= trainHt(find(predicted==labels(j)),j)+AlphaT(t);
end
[max_a,maxindex]=max(trainHt,[],2);
ResultHt=maxindex;
for j=1:length(labels)
    ResultHt(find(maxindex==j))= labels(j);
end

% For training epoch t = 2 : Max_Iter
for t = 2 : Max_Iter
    
    % 1): Train weak classifier nb using distribution weight
    nb = classregtree(traindata,trainlabel,'weights',weight,'method','classification');
    prec=eval(nb,traindata);

    % 2): get weak classifier
    predicted=cellfun(@str2num, prec);

    % 3): calculate the penalty value for every example
    for j=1:length(labels)
        trainHt(find(predicted==labels(j)),j)= trainHt(find(predicted==labels(j)),j)+AlphaT(t);
    end

    [max_a,maxindex]=max(trainHt,[],2);
    ResultHt=maxindex;
    for j=1:length(labels)
        ResultHt(find(maxindex==j))= labels(j);
    end
    
    for i=1:length(trainlabel)
        if ResultHt(i)~=predicted(i)
            if trainlabel(i)==predicted(i)
                amb(i)=amb(i)-1;
            else
                amb(i)=amb(i)+1;
            end
        end
    end

    ambt=amb/t;
    pt=1-ambt;
    ptlama=pt .^ lama;
    Pweight=weight .* ptlama;
    errorRate=sum(Pweight(find(predicted~=trainlabel)));
    rightRate=sum(Pweight(find(predicted==trainlabel)));

    % 4): calculate AlphaT by error and penalty
    AlphaT(t)=0.5*log((rightRate+eps)/(errorRate+eps));
    Learners{t} = nb;

    % 5): update data weights and obtain new weights by error and penalty
    for i=1:length(trainlabel)
        if trainlabel(i)==predicted(i)
            weight(i)=Pweight(i)* exp(- 1 * ( AlphaT(t)));
        else
            weight(i)=Pweight(i)* exp( AlphaT(t)*0) ;
        end
    end
    
    Z = sum(weight);
    weight = weight / Z;
    
end
time1=toc;
tic;

Result = zeros(size(testdata, 1),length(labels));
% Output the Final ensemble
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
