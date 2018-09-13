clear;
sname='9.satimage-5an-nn.mat';
load(sname);
kfold=5;
[a,b]=size(featuretest);

dup_all_whole_mat = featuretest;
dup_all_whole_labels = targettest;
c=floor(a*0.9);
traindata=dup_all_whole_mat(1:c,:);
trainlabel=dup_all_whole_labels(1:c,:);
testdata=dup_all_whole_mat(c+1:a,:);
testlabel=dup_all_whole_labels(c+1:a,:);

%直接使用cart
ft = classregtree(traindata,trainlabel,'method','classification');
prec=eval(ft,testdata);
prec=cellfun(@str2num, prec);
[kc,lratec,result2c,accc,gmeanc,result5c,fmeasurec] = calculateFunc(testlabel,prec);


%OVA
pre = classOVA(traindata,trainlabel,testdata);
[k1,lrate1,result21,acc1,gmean1,result51,fmeasure1] = calculateFunc(testlabel,pre);

%OVO
pre = classOAO([traindata,trainlabel],testdata);
[k2,lrate2,result22,acc2,gmean2,result52,fmeasure2] = calculateFunc(testlabel,pre);

%OAHO
pre = classOAHO([traindata,trainlabel],testdata);
[k3,lrate3,result23,acc3,gmean3,result53,fmeasure3] = calculateFunc(testlabel,pre');

%AandO
pre = classAandO(traindata,trainlabel,testdata);
[k4,lrate4,result24,acc4,gmean4,result54,fmeasure4] = calculateFunc(testlabel,pre');
