% Reference:	
% Name: funClassifierDECOC.m
% 
% Purpose: the ECOC matrix for all the classes is an nc*number1 matrix, each row represents the codeword of one class.
%          each column out of the number1 columns, will have a corresponding best classifier.
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

function [code,ft,labels,D] = funClassifierDECOC(traindata,trainlabel,type)

labels = unique(trainlabel);
nc = length(labels);
code = funECOCim(nc,type);

for i=1:nc
    idi=(trainlabel==labels(i));
    train{i}=traindata(idi,:);
    len(i)=length(trainlabel(idi));
end

numberl=size(code,2); % number1 represents the number of columns in the ECOC matrix: code
for t=1:numberl
    Dt=[];
    Dtlabel=[];
    flagDt=0;
    numberAp=0;
    numberAn=0;
    numberP=0;
    numberN=0;

    for i=1:nc
        Dt=[Dt;train{i}];
        Dtlabel(flagDt+1:flagDt+len(i))=code(i,t);
        flagDt=flagDt+len(i);

        if code(i,t)==1
            numberAp=numberAp+1;
            numberP=numberP+len(i);
        elseif code(i,t)==-1
            numberAn=numberAn+1;
            numberN=numberN+len(i);
        end
    end

    ct=[];
    flagct=0;
    for j=1:nc
        if code(j,t)==1
            cti=max(numberP,numberN)/(numberAp*len(j));
            ct(flagct+1:flagct+len(j))=cti;
            flagct=flagct+len(j);
        elseif code(j,t)==-1
            cti=max(numberP,numberN)/(numberAn*len(j));
            ct(flagct+1:flagct+len(j))=cti;
            flagct=flagct+len(j);
        end 
    end

  %  ft{t} = classregtree(Dt,Dtlabel,'weights',ct,'method','classification');

  %  find the best classifier for the current (transformed) binary data (using ECOC), i.e., [Dt,Dtlabel'].
  [ft{t,1},ft{t,2}] = bestClassifierEDOVO([Dt,Dtlabel'],5);

  D{t}=[Dt,Dtlabel'];
end
