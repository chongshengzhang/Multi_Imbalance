function runImECOCOVA
javaaddpath('weka.jar');

p = genpath(pwd);
addpath(p, '-begin');
% record = 'testall.txt';
% save record record



dataset_list = {'Wine_data_set_indx_fixed'};



for p = 1:length(dataset_list)%1:numel(dataset_list)
    load(['data\', dataset_list{p},'.mat']);
    disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);
    
    
    %imECOC+OVA
    for d=1:5
     
        [Cost(d).imECOCo1tr,Cost(d).imECOCo1te,Pre(d).imECOCo1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'OVA',1);
        
    end
    
    
    
    save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
    save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
    
    clear Cost Pre Indx;

end
