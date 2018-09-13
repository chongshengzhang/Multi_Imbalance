% This function first trains all the candidate classifiers , each of which is trained using a different
% classification algorithm. 8 different candidate algorithms are considered in our work, which are SVM, KNN,
% Logistic Regression, C4.5, AdaBoost, Random Forests, CART and Multilayer perceptron (MLP).

% it then calculates the validation error of each candidate classifier,
% and finds the one that corresponds to the minimum validation error (denoted as Cbest)

function [Cbest,bestk] = bestClassifier(train,kfold)

data = tokfold(train,kfold);

%%%%%%%%%%%%%%%libsvm 1
for i=1:kfold
    model = svmtrain(data(i).trainlabel, data(i).train,'-t 3 -q');
    [predict_label_L, accuracy_L, dec_values_L] = svmpredict(data(i).testlabel, data(i).test, model);
    
    AC=data(i).testlabel-predict_label_L;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
maxacc=mean(ACC);
Cbest=1;

%%%%%%%%%%%%%%%knn 2
acck=0;
bestk=3;
for k=3:2:11
    for i=1:kfold
        predict_label_knn=knnclassify(data(i).train,data(i).trainlabel,data(i).test,k,'cosine','random');
        AC=data(i).testlabel-predict_label_knn;
        AN=find(AC==0);
        ACC(i)=size(AN,1)/length(data(i).testlabel);
    end
    acc=mean(ACC);
    if acc>acck
        acck=acc;
        bestk=k;
    end    
end
if acck>maxacc
    maxacc=acck;
    Cbest=2;
end

%%%%%%%%%%%%%%%LR 3
for i=1:kfold
    [acc,predicted] = funLR(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    ACC(i)=acc;
end
acclr=mean(ACC);
if acclr>maxacc
    maxacc=acclr;
    Cbest=3;
end

%%%%%%%%%%%%%%%c45 4
for i=1:kfold
    [predictionlabel,acc] = func45(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    ACC(i)=acc;
end
acc45=mean(ACC);
if acc45>maxacc
    maxacc=acc45;
    Cbest=4;
end

%%%%%%%%%%%%%%%ada 5
for i=1:kfold
    [predictionlabelada]=adafunc(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    AC=data(i).testlabel-predictionlabelada;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
accada=mean(ACC);
if accada>maxacc
    maxacc=accada;
    Cbest=5;
end

%%%%%%%%%%%%%%%RF 6
for i=1:kfold
    B = m5pbuild(data(i).train, data(i).trainlabel);
    predict_label = m5ppredict(B,data(i).test); 
    predict_label_rf =  predict_label;
    AC=data(i).testlabel-predict_label_rf;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
accrf=mean(ACC);
if accrf>maxacc
    maxacc=accrf;
    Cbest=6;
end

%%%%%%%%%%%%%%%%%%%%%%%cart 7
for i=1:kfold
    ft=treefit(data(i).train, data(i).trainlabel,'method','classification');
    predictionlabelcart0=treeval(ft,data(i).test);
    predictionlabelcart = predictionlabelcart0;
    AC=data(i).testlabel-predictionlabelcart;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
acccart=mean(ACC);
if acccart>maxacc
    maxacc=acccart;
    Cbest=7;
end

%%%%%%%%%%%%%%%mlp 8
for i=1:kfold
    [predictionlabelmlp]=MLPfunc(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    AC=data(i).testlabel-predictionlabelmlp;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
accmlp=mean(ACC);
if accmlp>maxacc
    maxacc=accmlp;
    Cbest=8;
end
