%candidate algorithm Multilayer perceptron
function [output_sample_disc]=MLPfunc(feature_training,target_training,feature_test,target_test)



 Classificatore(1).Tipo = 'functions.MultilayerPerceptron';
 Classificatore(1).Parametri= {'-L'; '0.3'; '-M'; '0.2'; '-N'; '500'; '-V'; '20'; '-S'; '0'; '-E'; '20'; '-H'; 'a'};

 
Classificatore(1).nome_feature = crea_array_nome_feature ( size(feature_training, 2));

    
    %------------------------
    % training
    %------------------------

    target_training_cell = crea_array_celle_etichette(target_training);
    
    
    %Convert to weka format
    train = matlab2weka2('train',Classificatore(1).nome_feature,feature_training,target_training_cell);
    
    if isempty(Classificatore(1).Parametri)
        classifier_trained = trainWekaClassifier(train, Classificatore(1).Tipo);
    else    
        classifier_trained = trainWekaClassifier(train, Classificatore(1).Tipo, Classificatore(1).Parametri);
    end
    
    %------------------------
    % test
    %------------------------

    fake_target  = ones(size(feature_test,1),1);
    fake_target_cell = crea_array_celle_etichette(fake_target);
    test =  matlab2weka2('test',Classificatore(1).nome_feature,feature_test, fake_target_cell);

    [output_sample_disc, reliability_orig] = wekaClassify(test,classifier_trained);


