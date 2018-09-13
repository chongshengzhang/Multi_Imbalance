function prelabel = funcpreE(testdata,code,ft,W)
numbertest=size(testdata,1);
numberclass=size(code,1);
for t=1:length(ft)
    prec=treeval(ft{t},testdata);
    fX(:,t) = prec;
end
for i=1:numbertest
    ftx=fX(i,:);
    for r=1:numberclass
        for t=1:length(ftx)
            btr(t)=(1-ftx(t)*code(r,t))/2;
        end
        br=btr';
        yall(r)=W*br;
    end

    [minval,minindex]=min(yall);
    prelabel(i)=minindex;
end




