function runDECOC
javaaddpath('weka.jar');

p = genpath(pwd);
addpath(p, '-begin');
% record = 'testall.txt';
% save record record



dataset_list = {'Wine_data_set_indx_fixed'};



for p = 1:length(dataset_list)%1:numel(dataset_list)
    load(['data\', dataset_list{p},'.mat']);
    disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);
    
    
     %DECOC
    for d=1:5
  
        [Cost(d).imECOCDOVOs1tr,Cost(d).imECOCDOVOs1te,Pre(d).imECOCDOVOs1] = DECOC(data(d).train,data(d).trainlabel,data(d).test, 'sparse',1);
      
    end
    
    
    
    save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
    save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
    
    clear Cost Pre Indx;

end
