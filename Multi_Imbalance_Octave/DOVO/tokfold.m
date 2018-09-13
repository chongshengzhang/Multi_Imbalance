
function data = tokfold(train,kfold)
 target=train(:,end);
 
 feature1=train(:,1:end-1);
 

 cobj = cvpartition(target, 'kfold', kfold);
 

 for iter = 1:get(cobj,'NumTestSets')
  data(iter).train=feature1(training(cobj,iter),:);
  data(iter).trainlabel=target(training(cobj,iter));
  data(iter).test=feature1(test(cobj,iter),:);
  data(iter).testlabel=target(test(cobj,iter)); 
end

