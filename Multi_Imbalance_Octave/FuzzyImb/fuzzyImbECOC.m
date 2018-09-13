%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: E. Ramentol, S. Vluymans, N. Verbiest, et al. , IFROWANN: Imbalanced Fuzzy-Rough Ordered Weighted
%            Average Nearest Neighbor Classification, IEEE Transactions on Fuzzy Systems 23 (5) (2015) 1622-1637.
%
% Note 1: We obtain the codes of IFROWANN from the authors, we greatly acknowledge their help and contributions;
% Note 2: IFROWANN was originally designed for binary imbalanced data.
%         In this work, we extend IFROWANN with the ECOC encoding strategy to handle multi-class imbalanced data.
%
%(1) using funECOC(), it first generates the ECOC matrix (with each codeword for a specific class)
%    see lines 31-33; each class will be represented by an array of codes such as 1 1 -1 -1 1 -1.
%
%(2) it then extracts the instances (and the corresponding labels) of each original class, keep in train{i}
%    see lines 34-38;
%
%(3) the ECOC matrix for all the classes is an nc*number1 matrix, each row represents the codeword of one class.
%    a)  for each column of the ECOC matrix, first retrieve the corresponding bit value of each class.
%        then assign this bit value as the label for all the instances of the current class.
%        see lines 40-63; This is for handling the multi-class data.
%
%    b)  for each two-class data, use fuzzyImb to train the binary classifier (see above reference); see line 65.
%    at the end, we will train a few binary classifiers.
%
%    c)  for each test instance from testdata, use all the classifiers obtained from b) to make predictions;
%    see line 65, their predicitions will be  combined as an array,
%    then use the ECOC decoding method to find the nearest ECOC codeword, then the  corresponding class label.
%    see lines 72-89.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function prelabel=fuzzyImbECOC(traindata,trainlabel,testdata,testlabel,weightStrategy,gamma)

labels = unique (trainlabel);
numberc=length(labels);
code = funECOC(numberc);
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

    fX(:,t)=fuzzyImb(Dt,Dtlabel',testdata,testlabel,weightStrategy,gamma);
    
end

numbertest=size(testdata,1);
numberclass=size(code,1);

% below, it aims at the ECOC decoding on the testdata.
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
