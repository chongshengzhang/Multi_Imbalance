%apply Genetic Algorithm to search the optimum cost setup of each class
function C=GAtest(traindata,trainlabel)
global train;
train=[traindata,trainlabel];
labels=unique(train(:,end));
ObjectiveFunction = @GAfunc;
nvars = length(labels);    % Number of variables
for i=1:nvars
    LB(i) = 1e-5;  
end% Lower bound
UB = ones(1,nvars);  % Upper bound

[x,fval] = ga(ObjectiveFunction,nvars,[],[],[],[],LB,UB,[]);
x0=x/sum(x);
for i=1:length(trainlabel)
    indexc=find(labels==trainlabel(i));
    C(i)=x0(indexc);
end

    