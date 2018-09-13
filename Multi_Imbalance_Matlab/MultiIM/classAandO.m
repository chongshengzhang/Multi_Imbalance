% Reference:
% Ghanem, A. S., Venkatesh, S. & West, G. (2010). Multi-class pattern classification in imbalanced data.
% International Conference on Pattern Recognition, 2010 (PP. 2881-2884).
%
% PRMs-IM is a classification algorithm (originally designed) for binary imbalanced data.
% Let m be the ratio between the number of majority samples and that of the minority samples. PRMs-IM
% randomly divides the majority samples into m parts, next combines each part with all the minority
% instances, then trains a corresponding binary classifier.  In the prediction phase, it uses weighted voting
% to ensemble the outputs of the m classifiers and makes the final prediction.
%
% The above Reference proposes Multi-IM that
% combines A&O and PRMs-IM, where PRMs-IM is adopted to train the classifier for A&O.
%
% Besides A&O, in our work, we also combine the OVA, OVO and OAHO decomposition methods
% with PRMs-IM to further investigate the performance of PRMs-IM for multi-class imbalance learning.
%
% The following code is an implementation of Multi-IM+A&O
%
% Multi-IM+A&O
%
% The All-and-One (A&O) method combines the advantages of OVA and OVO and avoids their shortcomings.
% When predicting a new sample, A&O first uses OVA to get the top-2 prediction results (Ci,Cj),
% next adopts the OVO classifier previously trained for the pair of classes
% containing Ci and Cj to make the final prediction.
%

function [time1,time2,pre00] = classAandO(traindata,trainlabel,testdata)
tic;
%%%%%%%%%%%OVA
labels = unique (trainlabel);
numberc=length(labels);
codeA=zeros(numberc,numberc);
codeA(:,:)=-1;
for i=1:numberc
    codeA(i,i)=1;
end
for i=1:numberc
    idi=(trainlabel==labels(i));
    train{i}=traindata(idi,:);
    numbern(i)=length(trainlabel(idi));
end
numberl=numberc;
for t=1:numberl
    Dt=[];
    Dtlabel=[];
    flagDt=0;
    numberAp=0;
    numberAn=0;
    numberNp=0;
    numberNn=0;
    for i=1:numberc
        if codeA(i,t)==1
            Dt=[Dt;train{i}];
            Dtlabel(flagDt+1:flagDt+numbern(i))=1;
            flagDt=flagDt+numbern(i);
            numberAp=numberAp+1;
            numberNp=numberNp+numbern(i);
        elseif codeA(i,t)==-1
            Dt=[Dt;train{i}];
            Dtlabel(flagDt+1:flagDt+numbern(i))=-1;
            flagDt=flagDt+numbern(i);
            numberAn=numberAn+1;
            numberNn=numberNn+numbern(i);
        end
    end

    pre{t}= multiIMcart(Dt,Dtlabel',testdata);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%OVO
trainO=[traindata,trainlabel];
flagc=1;
for i=1:numberc-1
    for j=i+1:numberc
        idi=(trainO(:,end)==labels(i));
        idj=(trainO(:,end)==labels(j));
        Dij=[trainO(idi,:);trainO(idj,:)];

        CO{i,j} = multiIMcart(Dij(:,1:end-1),Dij(:,end),testdata);
        
        flagc=flagc+1;
    end
end



time1=toc;

tic;




numbertest=size(testdata,1);
numberC=size(codeA,1);
allpre=zeros(numbertest,numberC);
for t=1:length(pre)
    allpre(:,t)=pre{t};
end

for i=1:numbertest
    ftx=allpre(i,:);
    
    for r=1:numberC
        for t=1:length(ftx)
            btr(t)=(1-ftx(t)*codeA(r,t))/2;
        end
        yall(r)=sum(btr);
    end

    [b,index]=sort(yall);
    if index(1)<index(2)
        
        pre00(i)=CO{index(1),index(2)}(i);
    else
        pre00(i)=CO{index(2),index(1)}(i);
    end
end


time2=toc;
