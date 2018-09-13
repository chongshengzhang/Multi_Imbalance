function code = funOVAim(N_class)
code=zeros(N_class,N_class);
code(:,:)=-1;
for i=1:N_class
    code(i,i)=1;
end