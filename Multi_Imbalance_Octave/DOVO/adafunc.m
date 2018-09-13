%candidate algorithm AdaBoost
function [predicted] = adafunc(train,train_label,test,test_label)
test_label(1)=1;%%%%%%%%%%%%%%%%
test_label(2)=0;%%%%%%%%%%%%%%%%%%
feas=zeros(1,1);
feas=num2cell(feas);
feas{1,1}='class';
for i=1:size(train,2)
    str=['attr',num2str(i)];
    feas=[feas,str];
end
featureNames=feas(1,2:size(feas,2));
featureNames=[featureNames,'class'];

classindex = size(train,2)+1;

traindata=[num2cell(train),num2cell(train_label)];
testdata=[num2cell(test),num2cell(test_label)];

for i=1:size(traindata,1)
    traindata{i,size(traindata,2)}=['class',num2str(traindata{i,size(traindata,2)})];
end

for i=1:size(testdata,1)
    testdata{i,size(testdata,2)}=['class',num2str(testdata{i,size(testdata,2)})];
end

tr = matlab2weka('train',featureNames,traindata,classindex);
ts = matlab2weka('test',featureNames,testdata);

% train

nb = trainWekaClassifier(tr,'meta.AdaBoostM1');


% test

[predicted, classProbs] = wekaClassify(ts,nb);


