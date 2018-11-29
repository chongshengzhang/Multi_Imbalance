% Reference:	
% Name: funcwEDOVO.m
% 
% Purpose: on the training data, obtain the weight W for each dichotomy classifier,
%          trained using the ECOC decomposition strategy.
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

function W = funcwEDOVO(traindata,trainlabel,code,ft,labels,D)

numbertest=size(traindata,1);
W(1:length(ft))=sqrt(1/length(ft));

fX = funcPreEDOVO(traindata,trainlabel,ft,D);

% transform the label from 0 to -1, to be consistent with the ECOC codewords;
[a,b]=size(fX);
for i=1:a
    for j=1:b
        if fX(i,j)==0
            fX(i,j)=-1;
        end
    end
end
% fX(fX==0)=-1;

for i=1:length(labels)
    ny(i)=length(find(trainlabel==labels(i)));
end

for i=1:length(labels)
    gama(i)=max(ny)/ny(i);
end

% here uses the imECOC algorithm (decoding for ECOC)
% reference: Xu-Ying Liu et al. Learning Imbalanced Multi-class Data with Optimal Dichotomy Weights. IEEE ICDM 2013.
% see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf
% In the following, the goal is to obtain the weight for each dichotomy classifier.

for i=1:numbertest
    ftx=fX(i,:);
    indx=find(labels==trainlabel(i));
    yi=code(indx,:);

    for t=1:length(ftx)
        if ftx(t)~=yi(t)
            btyt=(1-ftx(t)*code(indx,t))/2;
            W(t)=W(t)+gama(indx)*btyt;
        end
    end
end


W=sqrt(W/sum(W));
