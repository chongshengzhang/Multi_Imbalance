
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