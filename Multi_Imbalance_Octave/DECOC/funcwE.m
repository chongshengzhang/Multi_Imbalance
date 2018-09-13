function W = funcwE(traindata,trainlabel,code,ft,labels)
numbertest=size(traindata,1);
%numberclass=size(code,1);
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
%    if yi~=ftx
        %   for r=1:numberclass
        %  if ftx(r)~=yi(r)
        for t=1:length(ftx)
            if ftx(t)~=yi(t)
                %    indx=find(labels==trainlabel(i));
                %        btr(t)=(1-ftx(t)*code(r,t))/2;
                btyt=(1-ftx(t)*code(indx,t))/2;
                W(t)=W(t)+gama(indx)*btyt;
                
                %                 br=btr';
                %                 by=bty';
                %             w=abs(trainlabel(i)-labels(r))/(br-by);
                
                %        yall(r)=W*br;
            end
        end
%     end
    
    %     [minval,minindex]=min(yall);
    %     prelabel(i)=minindex;
end
W=sqrt(W/sum(W));


