% Reference:
% Liu, X. Y., Li, Q. Q. & Zhou Z H. (2013). Learning imbalanced multi-class data with optimal dichotomy
% weights. IEEE 13th International Conference on Data Mining (IEEE ICDM), 2013 (PP.  478-487).
% see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf
%
% funcpre is the decoding function for the imECOC algorithm, on the test data;
% the goal is decoding (using the weight W obtained by the function funcw()), i.e., find the
% prediction result (array) is closest to which codeword, then output the corresponding original class label.
%
% step 14: for each test instance x, calculate the bit distance vector b(x, r), according to Eq. 4, 5, 6
% step 15: output the prelabel (Eq. 7)
%

function prelabel = funcpre(testdata,code,ft,W)
numbertest=size(testdata,1);
numberclass=size(code,1);

for t=1:length(ft)
    prec=eval(ft{t},testdata);
    fX(:,t)=cellfun(@str2num, prec);
end

for i=1:numbertest
    ftx=fX(i,:);
    for r=1:numberclass
        for t=1:length(ftx)
            btr(t)=(1-ftx(t)*code(r,t))/2;   % bit distance
        end
        br=btr';
        yall(r)=W*br;
    end

    [minval,minindex]=min(yall);
    prelabel(i)=minindex;                   % output prelabel
end
