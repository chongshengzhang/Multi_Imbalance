% Reference:	
% Name: funECOC.m
% 
% Authors: Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% Copyright: (c) 2018 Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% This file is a part of Multi_Imbalance software, a software package for multi-class Imbalance learning. 
% 
% Multi_Imbalance software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
% as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
%
% Multi_Imbalance software is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program. 
% If not, see <http://www.gnu.org/licenses/>.
function code = funECOC(N_class,type)

% For better explanation: 2^(N_class-1) and 3^N_class-2*2^N_class+1)/2 are the maximum length of codewords 
%(maximum number of nontrivial dichotomizers) in binary (dense) and ternary (sparse) coding design, respectively. 
% However, these numbers are unexpectedly large as the N_class increases. 
% So, Alwein et al. recommended 10*log2(N_class) and 15*log2(N_class) for
% dense and sparse ECOC, respectively. 

if nargin == 1
    type = 'sparse';
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
 