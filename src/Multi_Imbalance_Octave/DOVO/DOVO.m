% Reference:	
% Name: DOVO.m
% 
% Purpose: a multi-class classifier using one-against-one approach with different binary classifiers.
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
function [trainTime,testTime,pre,C] = DOVO(train,testdata,testlabel,kfold)
tic;
labels = unique (train(:,end));
nc=length(labels);
flagc=1;

%for each class pair (i,j)
for i=1:nc
    for j=i+1:nc
        idi=(train(:,end)==labels(i));
        idj=(train(:,end)==labels(j));

        % Dij is the set of data instances whose class labels are either i or j
        Dij=[train(idi,:);train(idj,:)];
        
        clabels = unique (Dij(:,end)); % there will be only two label values in clabels, i and j.
        ctrainlabel=Dij(:,end);        % the corresponding labels for all the instances in the training data

        for ci=1:length(ctrainlabel)   % transform class labels from (i ,j) to (0 , 1)
            if ctrainlabel(ci)==clabels(1)
                ctrainlabel(ci)=0;
            else
                ctrainlabel(ci)=1;
            end
        end
        train1=[Dij(:,1:end-1),ctrainlabel];
        Dij=train1;

        % find Cbest which is the best classifier that corresponds to the minimum validation error.
        % Cbest is the id of the classification algorithm chosen,
        % while bestk is the specific parameter needed by the KNN classifier
        [Cbest,bestk] = bestClassifier(Dij,kfold);

        % keep the current classifier information in D
        % --Cbest is the id of the classification algorithm chosen;
        % --bestk is the specific parameter needed by the KNN classifier;
        % --clables only contains two values, which are i and j
        % Note that, here, we only keep the id of the classification algorithm, but not the model trained
        % from the training dataset. The ideal situation is to keep the model as well (this should be fixed).
        D{flagc}=Dij;
        C{flagc,1}=Cbest;
        C{flagc,2}=bestk;
        L{flagc}=clabels;
        flagc=flagc+1;
    end
end
trainTime=toc;
tic;

pre = funcPre(testdata,testlabel,C,D,L); %test phase
testTime=toc;
%disp(Cbest);
