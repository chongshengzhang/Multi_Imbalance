% Reference:	
% Name: funcMLP.m
% 
% Purpose: candidate algorithm Multilayer perceptron
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

function [output_sample_disc]=funcMLP(feature_training,target_training,feature_test,target_test)



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


