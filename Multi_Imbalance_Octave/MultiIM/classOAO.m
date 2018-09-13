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
% The following code is an implementation of Multi-IM+OVO
%
% Multi-IM+OVO
%
% In the OVO approach, an independent binary classifier is
% built for each pair of classes. Thus, a classifier multiIMcart is trained
% using the samples of classes i and j, and hence this classifier
% is trained to discriminate between these two classes only.
% The simplest approach to combine the results of the OAO
% binary classifiers is majority voting, in which the test sample
% is assigned to the class with the highest number of votes.
%

function [time1,time2,pre0] = classOAO(train,testdata)
tic;
labels = unique (train(:,end));
numberc=length(labels);
flagc=1;
for i=1:numberc-1
    for j=i+1:numberc
        idi=(train(:,end)==labels(i));
        idj=(train(:,end)==labels(j));
        Dij=[train(idi,:);train(idj,:)];
        pre{flagc} = multiIMcart(Dij(:,1:end-1),Dij(:,end),testdata);
        flagc=flagc+1;
    end
end
time1=toc;

tic;
numbertest=size(testdata,1);
numberC=length(pre);
allpre=zeros(numbertest,numberC);
for t=1:length(pre)
    allpre(:,t)=pre{t};
end

pre0=mode(allpre,2);
time2=toc;
