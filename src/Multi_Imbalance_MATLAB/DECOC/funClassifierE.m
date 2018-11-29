% Reference:	
% Name: funClassifierE.m
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

function [code,ft,labels] = funClassifierE(traindata,trainlabel,type)
labels = unique (trainlabel);
numberc=length(labels);
code = funECOCim(numberc,type);
for i=1:numberc
    idi=(trainlabel==labels(i));
    train{i}=traindata(idi,:);
    numbern(i)=length(trainlabel(idi));
end
numberl=size(code,2);
for t=1:numberl
    Dt=[];
    Dtlabel=[];
    flagDt=0;
    numberAp=0;
    numberAn=0;
    numberNp=0;
    numberNn=0;
    for i=1:numberc
        if code(i,t)==1
            Dt=[Dt;train{i}];
            Dtlabel(flagDt+1:flagDt+numbern(i))=1;
            flagDt=flagDt+numbern(i);
            numberAp=numberAp+1;
            numberNp=numberNp+numbern(i);
        elseif code(i,t)==-1
            Dt=[Dt;train{i}];
            Dtlabel(flagDt+1:flagDt+numbern(i))=-1;
            flagDt=flagDt+numbern(i);
            numberAn=numberAn+1;
            numberNn=numberNn+numbern(i);
        end
    end
    ct=[];
    flagct=0;
    for i=1:numberc
        if code(i,t)==1
            cti=max(numberNp,numberNn)/(numberAp*numbern(i));
            ct(flagct+1:flagct+numbern(i))=cti;
            flagct=flagct+numbern(i);
        elseif code(i,t)==-1
            cti=max(numberNp,numberNn)/(numberAn*numbern(i));
            ct(flagct+1:flagct+numbern(i))=cti;
            flagct=flagct+numbern(i);
        end 
    end
    ft{t} = classregtree(Dt,Dtlabel,'weights',ct,'method','classification');
end
%yfit=eval(ft(t),X)

    
            