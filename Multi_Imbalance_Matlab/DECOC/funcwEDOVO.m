% on the training data, obtain the weight W for each dichotomy classifier,
% trained using the ECOC decomposition strategy.

function W = funcwEDOVO(traindata,trainlabel,code,ft,labels,D)

numbertest=size(traindata,1);
W(1:length(ft))=sqrt(1/length(ft));

fX = funcPreEDOVO(traindata,trainlabel,ft,D);

% transform the label from 0 to -1, to be consistent with the ECOC codewords;
[a,b]=size(fX);
for i=1:a
    for j=1:b
        if fX(i,j)==0
            fX(i,j)=-1;
        end
    end
end

for i=1:length(labels)
    ny(i)=length(find(trainlabel==labels(i)));
end

for i=1:length(labels)
    gama(i)=max(ny)/ny(i);
end

% here uses the imECOC algorithm (decoding for ECOC)
% reference: Xu-Ying Liu et al. Learning Imbalanced Multi-class Data with Optimal Dichotomy Weights. IEEE ICDM 2013.
% see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf
% In the following, the goal is to obtain the weight for each dichotomy classifier.

for i=1:numbertest
    ftx=fX(i,:);
    indx=find(labels==trainlabel(i));
    yi=code(indx,:);

    for t=1:length(ftx)
        if ftx(t)~=yi(t)
            btyt=(1-ftx(t)*code(indx,t))/2;
            W(t)=W(t)+gama(indx)*btyt;
        end
    end
end


W=sqrt(W/sum(W));
