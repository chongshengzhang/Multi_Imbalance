% Reference:	
% Name: bestClassifierEDOVO.m
% 
% Purpose: bestClassifierEDOVO iteratively tries 8 different classifiers to build a corresponding model,upon the training data (train)
%          It chooses the classifier that achieves the best accuracy results.
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

function [Cbest,bestk] = bestClassifierEDOVO(train,kfold)

% for i=1:size(train,1)
%     if train(:,end)==-1
%         train(:,end)=0;
%     end
% end
train(train(:,end)==-1)=0;

data = toKfold(train,kfold);

%the SVM classifier, libsvm 1
for i=1:kfold
    model = svmtrain(data(i).trainlabel, data(i).train,'-t 4 -q');
    [predict_label_L, accuracy_L, dec_values_L] = svmpredict(data(i).testlabel, data(i).test, model);
    
    AC=data(i).testlabel-predict_label_L;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
maxacc=mean(ACC);
Cbest=1;

%the KNN classifier, knn 2
acck=0;
bestk=3;
for k=3:2:11
    for i=1:kfold
        predict_label_knn=knnclassify(data(i).test,data(i).train,data(i).trainlabel,k,'cosine','random');
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

%the Logistic Regression Classifier, LR 3
for i=1:kfold
    [acc,predicted] = funLR(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    ACC(i)=acc;
end
acclr=mean(ACC);
if acclr>maxacc
    maxacc=acclr;
    Cbest=3;
end

%the C4.5 Classifier, c45 4
for i=1:kfold
    [predictionlabel,acc] = funC45(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    ACC(i)=acc;
end
acc45=mean(ACC);
if acc45>maxacc
    maxacc=acc45;
    Cbest=4;
end

%the AdaBoost Classifier, ada 5
for i=1:kfold
    [predictionlabelada]=funcAda(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    AC=data(i).testlabel-predictionlabelada;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
accada=mean(ACC);
if accada>maxacc
    maxacc=accada;
    Cbest=5;
end

%the Random Forest Classifier, RF 6
for i=1:kfold
    B = TreeBagger(100,data(i).train, data(i).trainlabel);
    predict_label = predict(B,data(i).test); 
    predict_label_rf = cellfun(@(x) str2double(x), predict_label);
    AC=data(i).testlabel-predict_label_rf;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
accrf=mean(ACC);
if accrf>maxacc
    maxacc=accrf;
    Cbest=6;
end

%the CART Classifier, cart 7
for i=1:kfold
    ft=classregtree(data(i).train, data(i).trainlabel,'method','classification');
    predictionlabelcart0=eval(ft,data(i).test);
    predictionlabelcart=cellfun(@str2num, predictionlabelcart0);
    AC=data(i).testlabel-predictionlabelcart;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
acccart=mean(ACC);
if acccart>maxacc
    maxacc=acccart;
    Cbest=7;
end

%the MLP Classifier, mlp 8
for i=1:kfold
    [predictionlabelmlp]=funcMLP(data(i).train, data(i).trainlabel, data(i).test, data(i).testlabel);
    AC=data(i).testlabel-predictionlabelmlp;
    AN=find(AC==0);
    ACC(i)=size(AN,1)/length(data(i).testlabel);
end
accmlp=mean(ACC);
if accmlp>maxacc
    maxacc=accmlp;
    Cbest=8;
end
