% calculate the accuracy of prelabel
function [k,lrate,result2,result3,result4,result5,result6] = calculateFunc(ACTUAL,PREDICTED)

label=unique(ACTUAL);
k=length(label);

for m=1:k

    idx = (ACTUAL()==label(m));
    
    p(m) = length(ACTUAL(idx));
    n(m) = length(ACTUAL(~idx));
    labelrate(m)=p(m)/length(ACTUAL);
    
    if p(m)~=0
        
        tp(m) = sum(ACTUAL(idx)==PREDICTED(idx));
        tn(m) = sum(ACTUAL(~idx)==PREDICTED(~idx));
        fp(m) = n(m)-tn(m);
        fn(m) = p(m)-tp(m);
        
        
            tp_rate(m) = tp(m)/p(m);
        
        
            tn_rate(m) = tn(m)/n(m);
        
        acc(m)=tp_rate(m);


        sensitivity(m) = tp_rate(m);
        
        
            precision(m) = tp(m)/(tp(m)+fp(m));
        
        recall(m) = sensitivity(m);
        
        
            f_measure(m) = 2*((precision(m)*recall(m))/(precision(m) + recall(m)));
        
    end
end

result2=[];
result3=mean(acc(~isnan(acc)));
result5=[];
result6=mean(f_measure(~isnan(f_measure)));
lrate=[];
result4=1;
nonan=0;
for m=1:k
    lrate=[lrate,num2str(labelrate(m)),','];
    if ~isnan(acc(m))
        result4=result4*acc(m);
        nonan=nonan+1;
    end
    result2=[result2,num2str(acc(m)),','];
    result5=[result5,num2str(f_measure(m)),','];
end
result4=result4^(1/nonan);
if isnan(result3)
    result3=num2str(result3);
end
if isnan(result4)
    result4=num2str(result4);
end
if isnan(result6)
    result6=num2str(result6);
end


