function gmean=GAfunc(x)
%testlabel=[1;2;3;1;2;3;1;1;2;3;1;1;1;2;2;3];
%prelabel=[1;2;3;2;2;3;1;3;2;2;1;1;1;2;2;3];
global train;
% global trainlabel;
labels=unique(train(:,end));
numberall=size(train,1);
numbertrain=floor(numberall*0.8);
train(randperm(numberall),:) = train;
traindata=train(1:numbertrain,1:end-1);
trainlabel=train(1:numbertrain,end);
testdata=train(numbertrain+1:end,1:end-1);
testlabel=train(numbertrain+1:end,end);
for i=1:length(trainlabel)
    indexc=find(labels==trainlabel(i));
    weight(i)=x(indexc);
end
ft = classregtree(traindata,trainlabel,'weights',weight,'method','classification');
prec=eval(ft,testdata);
prec=cellfun(@str2num, prec);
[kc,lratec,result2c,accc,gmeanc,result5c,fmeasurec] = calculateFunc(testlabel,prec);
gmean=1/gmeanc;