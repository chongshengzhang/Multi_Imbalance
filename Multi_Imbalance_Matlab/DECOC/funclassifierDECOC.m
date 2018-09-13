%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(1) Reference (we design and implement the following algorithm):
%   Jingjun Bi, Chongsheng Zhang*. (2018). An Empirical Comparison on State-of-the-art Multi-class Imbalance
%   Learning Algorithms and A New Diversified Ensemble Learning Scheme.
%   Knowledge-based Systems, 2018, Vol. XXX, pp. XXX.
%
%(2) using funECOCim(), it first generates the ECOC matrix (with each codeword for a specific class)
%    see lines 27-29; each class will be represented by an array of codes such as 1 1 -1 -1 1 -1.
%
%(3) it then extracts the instances (and the corresponding labels) of each original class, keep in train{i}
%    see lines 31-35;
%
%(4) the ECOC matrix for all the classes is an nc*number1 matrix, each row represents the codeword of one class.
%    a)  for each column of the ECOC matrix, first retrieve the corresponding bit value of each class.
%        then assign this bit value as the label for all the instances of the current class.
%        see lines 47-49;
%    b)  assign weights to instances of different classes for the current column,
%        which is set by the average length/ the length of the current class.
%        see lines 61-73; please note that these weights are not used here in the final training step. see c).
%    c)  find the best classifier for the [training data, encoded class values of the current column],
%        using the EDOVO algorithm (bestClassifierEDOVO). see line 78.
%    Finally, each column out of the number1 columns, will have a corresponding best classifier.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [code,ft,labels,D] = funclassifierDECOC(traindata,trainlabel,type)

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
