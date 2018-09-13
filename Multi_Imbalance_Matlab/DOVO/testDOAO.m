clear;
load_fspackage;
sname='Cardiotocography_ten_class_data_set_indx_fixed.mat';
load(sname);
kfold=5;
[a,b]=size(feature1);

dup_all_whole_mat = feature1;
dup_all_whole_labels = target;
c=floor(a*0.9);
traindata=dup_all_whole_mat(1:c,:);
trainlabel=dup_all_whole_labels(1:c,:);
testdata=dup_all_whole_mat(c+1:a,:);
testlabel=dup_all_whole_labels(c+1:a,:);
[pre,C] = DOAO([traindata,trainlabel],testdata,testlabel,kfold);
AC=testlabel-pre;
AN=find(AC==0);
ACC=size(AN,1)/length(testlabel);