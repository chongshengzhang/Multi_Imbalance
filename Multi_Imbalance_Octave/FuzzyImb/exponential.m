function w=exponential(p)

w=zeros(1,p);
if p<1024
    denom = 2^p - 1;
    for i=1:p
        w(i) = 2.0^(i-1) / denom;
    end
else
    w(p) = 0.5;
    for i=p-1:-1:1
        w(i) = w(i+1) * 0.5;
    end
end
