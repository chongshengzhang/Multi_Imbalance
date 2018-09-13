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
% The following code is an implementation of Multi-IM+OAHO
%
% Multi-IM+OAHO
%
% OAHO first sorts in descending order the class by the number of samples.
% Let the sorted class being {C1,C2,...,Ck},with C1 having the largest number of samples.
% Starting from C1 until Ck-1, OAHO sequentially makes the current class as ¡positive class¡
% and all the rest classes with lower ranks as ¡negative classes¡, then trains a binary classifier.
% Therefore, there will be k-1 binary classifiers in total.
% When predicting a new sample, the first classifier is used to predict it,
% if the prediction result is C1 then output C1 as the final result;
% otherwise, it switches to the second classifier to make the prediction,
% and so on, until the final prediction result is obtained.
%

function [time1,time2,pre0] = classOAHO(train,testdata)
tic;
TABLE = tabulate(train(:,end));
idxt=find(TABLE(:,2)>0);
TABLE=TABLE(idxt,:);
[b,index]=sort(TABLE(:,2)');
% maxlabel1=TABLE(index(end),1);
% maxlabel2=TABLE(index(end-1),1);
labels = TABLE(:,1);
numberc=length(labels);
flagc=1;
for i=numberc:-1:2
    maxlabel=TABLE(index(i),1);
    idi=(train(:,end)==maxlabel);
    Dij=train(idi,:);
    maxlabel2=TABLE(index(i-1),1);
    for j=i-1:-1:1
        maxlabeln=TABLE(index(j),1);

        idj=(train(:,end)== maxlabeln);
        Didj=train(idj,:);
        Didj(:,end)=maxlabel2;
        Dij=[Dij;Didj];       
    end
    
%     Cbest= bestClassifier(Dij,kfold);%%%%%%%%%%%%%%%%%%%%%%%%%class
    pre(:,flagc) = multiIMcart(Dij(:,1:end-1),Dij(:,end),testdata);

%     D{flagc}=Dij;
%     C{flagc}=Cbest;
    flagc=flagc+1;
end
time1=toc;

tic;
numbertest=size(testdata,1);

for i=1:numbertest
    for j=1:size(pre,2)
            pre0(i) = pre(i,j);%%%%%%%%%%%%%%%%%%pre
            if pre0(i)==TABLE(index(numberc-j+1),1)
                break;
            end
    end
end
time2=toc;

