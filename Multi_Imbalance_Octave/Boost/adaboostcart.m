
%     AdaBoost 
% 1: Initialize the weight Vector with uniform distribution
% 2: for t=1 to Max_Iter do
% 3:    Fit a classifier nb to the training data using weights 
% 4:    Compute weighted error: errorRate=sum(weight(find(predicted~=trainlabel)));
% 5:    Compute AlphaT=0.5*log((1-errorRate)/(errorRate+eps));
% 6:    Update weights weight(i)=weight(i)* exp(- 1 * ( AlphaT(t) * trainlabel(i)* predicted(i)));
% 7:    Re-normalize weight
% 8: end for
% 9: Output Final Classifier


function ResultR = adaboostcart(traindata,trainlabel,testdata,Max_Iter)

Learners = {};
weight = ones(1, length(trainlabel)) / length(trainlabel);%step 1
AlphaT = zeros(1, Max_Iter);

for t = 1 : Max_Iter%step 2-8
    
    nb = treefit(traindata,trainlabel,'weights',weight,'method','classification');%step 3
    prec=treeval(nb,traindata);
    predicted = prec;
    
    errorRate=sum(weight(find(predicted~=trainlabel)));%step 4
    AlphaT(t)=0.5*log((1-errorRate)/(errorRate+eps));%step 5
    
    Learners{t} = nb;
    for i=1:length(trainlabel)%step 6
        weight(i)=weight(i)* exp(- 1 * ( AlphaT(t) * trainlabel(i)* predicted(i)));
    end
    
    Z = sum(weight);
    weight = weight / Z;%step 7
    
end
%step 9
Result = zeros(size(testdata, 1),1);

for i = 1 : length(Learners)
    
    lrn_out =  treeval(Learners{i}, testdata);
    lrn_out = lrn_out;
    Result = Result + lrn_out * AlphaT(i);
end

ResultR = sign(Result);