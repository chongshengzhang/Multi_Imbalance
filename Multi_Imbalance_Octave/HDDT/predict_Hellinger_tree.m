function predicted_classes = predict_Hellinger_tree(model,features)
%Function: predict_Hellinger_tree
%Form: predicted_classes = predict_Hellinger_tree(model,features)
%Description: Predict labels using trained Hellinger Distance Decision Tree
%Parameters:
%   model: a trained Hellinger Distance Decision Tree model
%   features: I X F numeric matrix where I is the number of instances and F
%       is the number of features. Each row represents one training instance
%       and each column represents the value of one of its corresponding features
%Output:
%   predicted_classes: I X 1 matrix where each row represents a predicted label of the corresponding feature set 

[numInstances,numFeatures] = size(features);

if numInstances <= 0
    msgID = 'predict_Hellinger_tree:notEnoughData';
    msg = 'Feature array is empty or only instance exists';
    causeException = MException(msgID,msg);
    throw(causeException);
end

if numFeatures == 0
    msgID = 'predict_Hellinger_tree:noData';
    msg = 'No feature data';
    causeException = MException(msgID,msg);
    throw(causeException);
end

initialModel = model;
predicted_classes = zeros(size(features,1),1);
for i = 1:1:size(features,1)
    model = initialModel;
    complete = model.complete;
    while ~complete
        if features(i,model.feature) <= model.threshold
            model = model.leftBranch;
        else
            model = model.rightBranch;
        end
        complete = model.complete;
    end
    predicted_classes(i) = model.label;
end
end