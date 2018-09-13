function w=additive_gamma(n,p,gamma)

r = ceil(p + gamma * (n - p));
w = zeros(1,n);
denom = r * (r + 1.0);
for i=n-r+1:n
    w(i) = (2.0 * ((i - (n-r)))) / denom;
end