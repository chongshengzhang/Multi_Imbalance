function code = funECOC(N_class,type)

% For better explanation: 2^(N_class-1) and 3^N_class-2*2^N_class+1)/2 are the maximum length of codewords 
%(maximum number of nontrivial dichotomizers) in binary (dense) and ternary (sparse) coding design, respectively. 
% However, these numbers are unexpectedly large as the N_class increases. 
% So, Alwein et al. recommended 10*log2(N_class) and 15*log2(N_class) for
% dense and sparse ECOC, respectively. 

if nargin == 1
    type = 'dense';
end

if strcmp(type , 'OVA')
%% Generate binary (OVA) ECOC matrix
    code = funOVA(N_class);
end

if strcmp(type , 'dense')
%% Generate binary (dense) ECOC matrix
    N_dichotomizers=min(2^(N_class-1)-1,floor(10*log2(N_class)));
    zero_prob=0.0;
    code =Pseudo_randdom_Coding(N_class,N_dichotomizers,zero_prob);
elseif strcmp(type , 'sparse')

%% Generate ternary (sparse) ECOC matrix
    N_dichotomizers=min((3^N_class-2*2^N_class+1)/2,floor(15*log2(N_class)));
    zero_prob=0.5; %it may be 0.5;
    code=Pseudo_randdom_Coding(N_class,N_dichotomizers,zero_prob);

end
 