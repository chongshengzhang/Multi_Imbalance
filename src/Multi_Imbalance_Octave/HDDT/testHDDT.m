% Reference:	
% Name: testHDDT.m
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
clear;


load('6alrawdata.mat');
label=unique(trainlabel);

for i = 1:length(label)
    trainingLabels = trainlabel == label(i);
    trainingFeatures = traindata;
    testLabels = testlabel == label(i);
    testFeatures = testdata;
    
    %Dataset statistics
    disp('Dataset: ')
    disp(['Number of Training Instances: ' num2str(size(trainingFeatures,1))]);
    disp(['Number of Test Instances: ' num2str(size(testFeatures,1))]);
    disp(['Number of Features (Measurements): ' num2str(size(trainingFeatures,2))]);
    disp(' ');
    
    %Run classifiers
    
    %HellingerTree
    disp('Hellinger Tree:')
    tic();
    model = fit_Hellinger_tree(trainingFeatures,trainingLabels);
    trainingTime = toc();
    tic();
    predictions = predict_Hellinger_tree(model,testFeatures);
    testTime = toc();
    correct = (predictions == testLabels);
    correct = sum(correct) / length(correct);
    disp(['Percent of instances correctly classified: ' num2str(correct)]);
    disp(['Training time: ' num2str(trainingTime) ' seconds']);
    disp(['Test time: ' num2str(testTime) ' seconds']);
    disp(['Total time: ' num2str(trainingTime + testTime) ' seconds']);
    disp(' ');
    
    
    
    %Linear SVM for Comparison
    disp('SVM:')
    tic();
    model = fitcsvm(trainingFeatures,trainingLabels);
    trainingTime = toc();
    tic();
    predictions = predict(model,testFeatures);
    testTime = toc();
    correct = (predictions == testLabels);
    correct = sum(correct) / length(correct);
    disp(['Percent of instances correctly classified: ' num2str(correct)]);
    disp(['Training time: ' num2str(trainingTime) ' seconds']);
    disp(['Test time: ' num2str(testTime) ' seconds']);
    disp(['Total time: ' num2str(trainingTime + testTime) ' seconds']);
    disp(' ');
    
end