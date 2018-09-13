function code = funOVOim(N_class)
% code=zeros(N_class,N_class);
cnum=N_class*(N_class-1)/2;
code=zeros(N_class,cnum);
flag=1;
for i=1:N_class-1
    for j=i+1:N_class
        code(i,flag)=1;
        code(j,flag)=-1;
        flag=flag+1;
    end
end