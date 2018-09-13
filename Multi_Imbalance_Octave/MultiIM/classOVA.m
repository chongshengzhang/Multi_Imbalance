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
% The following code is an implementation of Multi-IM+OVA
%
% Multi-IM+OVA
%
% In the Multi-IM+OVA approach, c binary classifiers are constructed,
% in which a classifier is constructed for each class.
% Thus, the classifier multiIMcart is trained using the samples of class
% Ci against all the samples of the other classes. The results
% of the binary classifiers can be combined using a decision function: F(x),
% which assigns the test sample to the class with the highest output value.
%

function [time1,time2,prelabel] = classOVA(traindata,trainlabel,testdata)
tic;
labels = unique (trainlabel);
numberc=length(labels);
code=zeros(numberc,numberc);
code(:,:)=-1;
for i=1:numberc
    code(i,i)=1;
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
   pre{t} = multiIMcart(Dt,Dtlabel',testdata);
end
time1=toc;

tic;
numbertest=size(testdata,1);
numberC=size(code,1);
allpre=zeros(numbertest,numberC);
for t=1:length(pre)
    allpre(:,t)=pre{t};
end

for i=1:numbertest
    ftx=allpre(i,:);
    
    for r=1:numberC
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
