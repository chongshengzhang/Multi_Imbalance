function [code,ft,labels] = funclassifierE(traindata,trainlabel,type)
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

    
            