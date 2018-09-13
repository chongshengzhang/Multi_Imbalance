function code = funDecompose(N_class,type)


if nargin == 1
    type = 'OVA';
end

if strcmp(type , 'OVA')
%% Generate binary (OVA) ECOC matrix
    code = funOVA(N_class);
end

if strcmp(type , 'OVO')

    N_dichotomizers=min(2^(N_class-1)-1,floor(10*log2(N_class)));
    zero_prob=0.0;
    code =Pseudo_randdom_Coding(N_class,N_dichotomizers,zero_prob);
end

if strcmp(type , 'OAHO')

    N_dichotomizers=min((3^N_class-2*2^N_class+1)/2,floor(15*log2(N_class)));
    zero_prob=0.5; %it may be 0.5;
    code=Pseudo_randdom_Coding(N_class,N_dichotomizers,zero_prob);

end

if strcmp(type , 'A&O')
%% Generate binary (OVA) ECOC matrix
    code = funOVA(N_class);
end
 