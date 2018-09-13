
function data = tokfold(train,kfold)
 target=train(:,end);
 
 feature1=train(:,1:end-1);
 

 cobj = cvpartition(target, 'kfold', kfold);
 
 

for iter = 1:cobj.NumTestSets
  data(iter).train=feature1(cobj.training(iter),:);
  data(iter).trainlabel=target(cobj.training(iter));
  data(iter).test=feature1(cobj.test(iter),:);
  data(iter).testlabel=target(cobj.test(iter)); 
 end

