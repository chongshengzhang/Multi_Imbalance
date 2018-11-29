% Reference:	
% Name: testAll.m
% 
% Purpose: an example of all methods tested
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
function testAll

javaaddpath('weka.jar');

p = genpath(pwd);
addpath(p, '-begin');
% record = 'testall.txt';
% save record record


%   'thyroiddis','Wine_data_set_indx_fixed','winequalitywhite'

dataset_list = {'Wine_data_set_indx_fixed'};


for p = 1:length(dataset_list)%1:numel(dataset_list)
    load(['data\', dataset_list{p},'.mat']);
    disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);
    
    %     funzione_n = 'DOAO';
    %     try
    
    
    %DOVO
    %input: traindata,trainlabel,testdata,testlabel,kfold
    %output: trainCostTime,predictCostTime,predictResult,bestChosen
    for d=1:5

        [Cost(d).DOAOtr,Cost(d).DOAOte,Pre(d).DOAO,Indx(d).C] = DOVO([data(d).train,data(d).trainlabel],data(d).test,data(d).testlabel,5);

    end
    
    %HDDT+ECOC
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5

        [Cost(d).HDDTecoctr,Cost(d).HDDTecocte,Pre(d).HDDTecoc] = HDDTECOC(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);

    end
    
    %HDDT+OVA
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
    
        [Cost(d).HDDTovatr,Cost(d).HDDTovate,Pre(d).HDDTova] = HDDTOVA(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);
        
    end
    
    %MC-HDDT
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
  
        [Cost(d).MCHDDTtr,Cost(d).MCHDDTte,Pre(d).MCHDDT] = MCHDDT(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);
    
    end
    
    
    %imECOC+sparse
    %input: traindata,trainlabel,testdata,testlabel,type,withw
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
    
        [Cost(d).imECOCs1tr,Cost(d).imECOCs1te,Pre(d).imECOCs1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'sparse',1);
       
    end
    
    %imECOC+OVA
    %input: traindata,trainlabel,testdata,testlabel,type,withw
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
     
        [Cost(d).imECOCo1tr,Cost(d).imECOCo1te,Pre(d).imECOCo1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'OVA',1);
        
    end
    
    %imECOC+dense
    %input: traindata,trainlabel,testdata,testlabel,type,withw
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
      
        [Cost(d).imECOCd1tr,Cost(d).imECOCd1te,Pre(d).imECOCd1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'dense',1);
    
    end
    
    %cart
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
        tic;
        ft = classregtree(data(d).train,data(d).trainlabel,'method','classification');
        Cost(d).carttr=toc;
        tic;
        prec=eval(ft,data(d).test);
        Pre(d).cart=cellfun(@str2num, prec);
        Cost(d).cartte=toc;
    end
    
    %Multi-IM+OVA
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
     
        [Cost(d).classOVAtr,Cost(d).classOVAte,Pre(d).classOVA] = classOVA(data(d).train,data(d).trainlabel,data(d).test);
     
    end
    
    %Multi-IM+OVO
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
      
        [Cost(d).classOAOtr,Cost(d).classOAOte,Pre(d).classOAO] = classOAO([data(d).train,data(d).trainlabel],data(d).test);
    
    end
    
    %Multi-IM+OAHO
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5

        [Cost(d).classOAHOtr,Cost(d).classOAHOte,Pre(d).classOAHO] = classOAHO([data(d).train,data(d).trainlabel],data(d).test);
      
    end
    
    %Multi-IM+A&O
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
       
        [Cost(d).classAandOtr,Cost(d).classAandOte,Pre(d).classAandO] = classAandO(data(d).train,data(d).trainlabel,data(d).test);
      
    end
    
    %AdaBoost.M1
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
   
        [Cost(d).adaboostcartM1tr,Cost(d).adaboostcartM1te,Pre(d).adaboostcartM1] = adaBoostCartM1(data(d).train,data(d).trainlabel,data(d).test,20);
     
    end
    
    %SAMME
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
       
        [Cost(d).SAMMEcarttr,Cost(d).SAMMEcartte,Pre(d).SAMMEcart] = sammeCart(data(d).train,data(d).trainlabel,data(d).test,20);
      
    end
    
    %AdaBoost.NC
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter,lama
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
 
        [Cost(d).adaboostcartNCtr,Cost(d).adaboostcartNCte,Pre(d).adaboostcartNC] = adaBoostCartNC(data(d).train,data(d).trainlabel,data(d).test,20,2);
    
    end
    
    %PIBoost
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
     
        [Cost(d).PIBoostcarttr,Cost(d).PIBoostcartte,Pre(d).PIBoostcart] = PIBoostCart(data(d).train,data(d).trainlabel,data(d).test,20);
      
    end
    
    %AdaC2.M1
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter,C
    %output: trainCostTime,predictCostTime,predictResult
    for d=1:5
        tic;
        C0=testGA(data(d).train,data(d).trainlabel);
        Cost(d).GA=toc;
        Indx(d).GA=C0;
       
        [Cost(d).adaC2cartM1GAtr,Cost(d).adaC2cartM1GAte,Pre(d).adaC2cartM1GA] = adaC2CartM1(data(d).train,data(d).trainlabel,data(d).test,20,C0);
     
    end
    
    %FuzzyImb+ECOC
    %input: traindata,trainlabel,testdata,testlabel,weightStrategy,gamma
    %output: trainCostTime,predictCostTime,predictResult
     for d=1:5
        tic;
        [Pre(d).fuzzyw6] = fuzzyImbECOC(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel, 'w6',0.1);
        Cost(d).fuzzyw6=toc;
     end
    
     %DECOC
     %input: traindata,trainlabel,testdata,testlabel,type,withw
     %output: trainCostTime,predictCostTime,predictResult
     for d=1:5
  
         [Cost(d).imECOCDOVOs1tr,Cost(d).imECOCDOVOs1te,Pre(d).imECOCDOVOs1] = DECOC(data(d).train,data(d).trainlabel,data(d).test, 'sparse',1);
      
     end
    
    
    
    save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
    save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
    
    clear Cost Pre Indx;

end
