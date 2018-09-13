clear;
sname='9.satimage-5an-nn.mat';
load(sname);
kfold=5;


dup_all_whole_mat = featuretest;
dup_all_whole_labels = targettest;
[a,b]=size(dup_all_whole_mat);
c=floor(a*0.9);
traindata=dup_all_whole_mat(1:c,:);
trainlabel=dup_all_whole_labels(1:c,:);
testdata=dup_all_whole_mat(c+1:a,:);
testlabel=dup_all_whole_labels(c+1:a,:);

%cart
ft = classregtree(traindata,trainlabel,'method','classification');
%ft = classregtree(traindata,trainlabel);
%prec=predict(ft,testdata);
prec=eval(ft,testdata);
prec=cellfun(@str2num, prec);
%prec=round(prec);
[kc,lratec,result2c,accc,gmeanc,result5c,fmeasurec] = calculateFunc(testlabel,prec);
% ACc=testlabel-prec;
% ANc=find(ACc==0);
% ACCc=size(ANc,1)/length(testlabel);
% 
pre = imECOC(traindata,trainlabel,testdata, 'sparse',0);%'dense''OVA''sparse'
pre=round(pre);
[k,lrate,result2,acc,gmean,result5,fmeasure] = calculateFunc(testlabel,pre');
% AC=testlabel-pre';
% AN=find(AC==0);
% ACC=size(AN,1)/length(testlabel);

%with w
pre = imECOC(traindata,trainlabel,testdata, 'sparse',1);%'dense''OVA''sparse'
pre=round(pre);
[kw,lratew,result2w,accw,gmeanw,result5w,fmeasurew] = calculateFunc(testlabel,pre');
% AC=testlabel-pre';
% AN=find(AC==0);
% ACC=size(AN,1)/length(testlabel);