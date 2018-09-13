function runAdaC2M1
javaaddpath('weka.jar');

p = genpath(pwd);
addpath(p, '-begin');
% record = 'testall.txt';
% save record record



dataset_list = {'Wine_data_set_indx_fixed'};



for p = 1:length(dataset_list)%1:numel(dataset_list)
    load(['data\', dataset_list{p},'.mat']);
    disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);
    
    
    %AdaC2.M1
    for d=1:5
        tic;
        C0=GAtest(data(d).train,data(d).trainlabel);
        Cost(d).GA=toc;
        Indx(d).GA=C0;
       
        [Cost(d).adaC2cartM1GAtr,Cost(d).adaC2cartM1GAte,Pre(d).adaC2cartM1GA] = adaC2cartM1(data(d).train,data(d).trainlabel,data(d).test,20,C0);
     
    end
    
    
    
    save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
    save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
    
    clear Cost Pre Indx;

end
