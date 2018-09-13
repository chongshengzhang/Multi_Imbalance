% Reference:
% Liu, X. Y., Li, Q. Q. & Zhou Z H. (2013). Learning imbalanced multi-class data with optimal dichotomy
% weights. IEEE 13th International Conference on Data Mining (IEEE ICDM), 2013 (PP.  478-487).
% see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf
%
% step 11 calculates the bit distance vector for the training set according to Eq. 4, 5, 6
% step 12 obtains the optimal dichotomy weights w according to Eq. 9

function W = funcw(traindata,trainlabel,code,ft,labels)

numbertest=size(traindata,1);

W(1:length(ft))=sqrt(1/length(ft));
for t=1:length(ft)
    prec=treeval(ft{t},traindata);
    fX(:,t) = prec;
end

for i=1:length(labels)
    ny(i)=length(find(trainlabel==labels(i)));
end

for i=1:length(labels)
    gama(i)=max(ny)/ny(i);
end

for i=1:numbertest
    ftx=fX(i,:);
    indx=find(labels==trainlabel(i));
    yi=code(indx,:);
    for t=1:length(ftx)
        if ftx(t)~=yi(t)
            btyt=(1-ftx(t)*code(indx,t))/2;    % step 11
            W(t)=W(t)+gama(indx)*btyt;         % step 12
        end
    end
end

W=sqrt(W/sum(W));
