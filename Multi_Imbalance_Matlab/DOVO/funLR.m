%candidate algorithm Logistic Regression
function [acc,predicted] = funLR(train, train_label, test, test_label)

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

nb = trainWekaClassifier(tr,'functions.Logistic');
predicted = wekaClassify(ts,nb);
actual = ts.attributeToDoubleArray(classindex-1);


acc = sum(actual == predicted)/size(test,1);