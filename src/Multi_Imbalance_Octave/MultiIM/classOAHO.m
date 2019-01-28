% Reference:	
% Name: classOAHO.m
% 
% Purpose: a classification algorithm (originally designed) for binary imbalanced data.
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
function [trainTime,testTime,preResult] = classOAHO(train,testdata)
tic;
TABLE = tabulate(train(:,end));
idxt=find(TABLE(:,2)>0);
TABLE=TABLE(idxt,:);
[b,index]=sort(TABLE(:,2)');
% maxlabel1=TABLE(index(end),1);
% maxlabel2=TABLE(index(end-1),1);
labels = TABLE(:,1);
numberc=length(labels);
flagc=1;
for i=numberc:-1:2
    maxlabel=TABLE(index(i),1);
    idi=(train(:,end)==maxlabel);
    Dij=train(idi,:);
    maxlabel2=TABLE(index(i-1),1);
    for j=i-1:-1:1
        maxlabeln=TABLE(index(j),1);

        idj=(train(:,end)== maxlabeln);
        Didj=train(idj,:);
        Didj(:,end)=maxlabel2;
        Dij=[Dij;Didj];       
    end
    
%     Cbest= bestClassifier(Dij,kfold);%%%%%%%%%%%%%%%%%%%%%%%%%class
    pre(:,flagc) = multiIMCart(Dij(:,1:end-1),Dij(:,end),testdata);

%     D{flagc}=Dij;
%     C{flagc}=Cbest;
    flagc=flagc+1;
end
trainTime=toc;

tic;
numbertest=size(testdata,1);

for i=1:numbertest
    for j=1:size(pre,2)
            preResult(i) = pre(i,j);%%%%%%%%%%%%%%%%%%pre
            if preResult(i)==TABLE(index(numberc-j+1),1)
                break;
            end
    end
end
testTime=toc;

