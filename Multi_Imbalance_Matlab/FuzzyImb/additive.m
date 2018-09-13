function w=additive(n)

w=zeros(1,n);
denom =  n * (n + 1.0); 
for i = 1:n
    w(i) = (2.0 * i) / denom;
end