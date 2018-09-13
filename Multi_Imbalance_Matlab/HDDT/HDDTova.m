% Reference：
% Hoens, T. R., Qian, Q., Chawla, N. V., et al. (2012). Building decision trees for the multi-class imbalance
% problem. Advances in Knowledge Discovery and Data Mining. Springer Berlin Heidelberg, 2012 (PP. 122-134).
%
% Decomposition Techniques OVA + HDDT, for multi-class imbalanced data.
% This is our own extension of HDDT to multi-class imbalanced data.
% it builds numberc of binary HDDT classifiers by combining the OVA strategy and HDDT, see lines 25-52.
% then combines the outputs of different binary HDDT classifiers generated using the OVA strategy,
% see lines 59-80, here, the decoding strategy for OVA is the same as the imECOC decoding. 
% It must be noted that the decoding strategy for testHDDTecoc and testHDDTova are identical,
% for fair comparisons between them.

function [time1,time2,prelabel]=HDDTova(traindata,trainlabel,testdata,testlabel)
tic;
labels = unique (trainlabel);
numberc=length(labels);
code = funOVA(numberc);

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
            Dtlabel(flagDt+1:flagDt+numbern(i))=0;
            flagDt=flagDt+numbern(i);
            numberAn=numberAn+1;
            numberNn=numberNn+numbern(i);
        end
    end
    
    ft{t} = fit_Hellinger_tree(Dt,Dtlabel');
end
                               
time1=toc;
                               
tic;
numbertest=size(testdata,1);
numberclass=size(code,1);
for t=1:length(ft)
    fX(:,t)=predict_Hellinger_tree(ft{t},testdata);
end
                               
for i=1:numbertest
    ftx=fX(i,:);
    for t=1:length(ftx)
        if ftx(t)==0
            ftx(t)=-1;
        end
    end
   
    for r=1:numberclass
        for t=1:length(ftx)
            btr(t)=(1-ftx(t)*code(r,t))/2;
        end
        yall(r)=sum(btr);
    end

    [minval,minindex]=min(yall);
    prelabel(i)=labels(minindex);
end

prelabel=prelabel';
time2=toc;
