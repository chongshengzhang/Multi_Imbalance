function runCART
javaaddpath('weka.jar');

p = genpath(pwd);
addpath(p, '-begin');
% record = 'testall.txt';
% save record record



dataset_list = {'Wine_data_set_indx_fixed'};



for p = 1:length(dataset_list)%1:numel(dataset_list)
    load(['data\', dataset_list{p},'.mat']);
    disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);
    
    
    %cart
    for d=1:5
        tic;
        ft = treefit(data(d).train,data(d).trainlabel,'method','classification');
        Cost(d).carttr=toc;
        tic;
        prec=treeval(ft,data(d).test);
        Pre(d).cart = prec;
        Cost(d).cartte=toc;
    end
    
    
    
    save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
    save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
    
    clear Cost Pre Indx;

end
