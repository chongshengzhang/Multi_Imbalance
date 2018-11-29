% Reference:	
% Name: funcPre.m
% 
% Purpose: the classification of test datapoints is realised by combining the decisions of individual classifiers.
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

function pre = funPre(testdata,testlabel,C,D,L)
[numberC,a]=size(C);
numbertest=length(testlabel);
allpre=zeros(numbertest,numberC);
for i=1:numberC
    train=D{i};

    testlabel(:)=max(train(:,end));
    testlabel(2)=min(train(:,end));
    if C{i,1}==1
        model = svmtrain(train(:,end), train(:,1:end-1),'-t 4 -q');
        [predict_label_L, accuracy_L, dec_values_L] = svmpredict(testlabel, testdata, model);
        allpre(:,i)=predict_label_L;
    end
    if C{i,1}==2
        predict_label_L=knnclassify(testdata,train(:,1:end-1), train(:,end),C{i,2},'cosine','random');
        allpre(:,i)=predict_label_L;
    end
    if C{i,1}==3
        [acc,predicted] = funLR(train(:,1:end-1), train(:,end), testdata,testlabel);
        allpre(:,i)=predicted;
    end
    if C{i,1}==4
        [predictionlabel,acc] = funC45(train(:,1:end-1), train(:,end), testdata,testlabel);
        
        allpre(:,i)=predictionlabel;
    end
    if C{i,1}==5
        [predictionlabel]=funcAda(train(:,1:end-1), train(:,end), testdata,testlabel);
        allpre(:,i)=predictionlabel;
    end
    if C{i,1}==6
        B = TreeBagger(100,train(:,1:end-1), train(:,end));
        predict_label = predict(B,testdata);
        predict_label_rf = cellfun(@(x) str2double(x), predict_label);
        allpre(:,i)=predict_label_rf;
    end
    if C{i,1}==7
        ft=classregtree(train(:,1:end-1), train(:,end),'method','classification');
        predictionlabelcart0=eval(ft,testdata);
        predictionlabelcart=cellfun(@str2num, predictionlabelcart0);
        allpre(:,i)=predictionlabelcart;
    end
    if C{i,1}==8
        [predictionlabel]=funcMLP(train(:,1:end-1), train(:,end), testdata,testlabel);
        allpre(:,i)=predictionlabel;
    end
    label=L{i};
    for j=1:length(testlabel)
        if allpre(j,i)==0
            allpre(j,i)=label(1);
        else
            allpre(j,i)=label(2);
        end
    end
end
pre=mode(allpre,2);
