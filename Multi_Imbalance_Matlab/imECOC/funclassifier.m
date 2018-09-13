% Reference:
% Liu, X. Y., Li, Q. Q. & Zhou Z H. (2013). Learning imbalanced multi-class data with optimal dichotomy
% weights. IEEE 13th International Conference on Data Mining (IEEE ICDM), 2013 (PP.  478-487).
% see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf
%


function [code,ft,labels] = funclassifier(traindata,trainlabel,type)

%% step 1-10 of the imECOC algorithm in the above reference.

labels = unique (trainlabel);
numberc=length(labels);
code = funECOCim(numberc,type);  % step 1: generate the code matrix
for i=1:numberc
    idi=(trainlabel==labels(i));
    train{i}=traindata(idi,:);
    numbern(i)=length(trainlabel(idi));
end

numberl=size(code,2);
for t=1:numberl                  % steps 2-10 of the imECOC algorithm
    Dt=[];                       % step 3 of the imECOC algorithm
    Dtlabel=[];
    flagDt=0;
    numberAp=0;
    numberAn=0;
    numberNp=0;
    numberNn=0;

    for i=1:numberc               % steps 4-8 of the imECOC algorithm
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

    % step 9: calculate the example weights ct for the t-th dichotomy according to Eq. 2.
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

    % step 10: learn a classifier using the weighted examples
    ft{t} = classregtree(Dt,Dtlabel,'weights',ct,'method','classification');
end
%yfit=eval(ft(t),X)
