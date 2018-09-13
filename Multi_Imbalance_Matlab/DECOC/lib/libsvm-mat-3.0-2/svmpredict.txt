function [predicted_label, accuracy, dval_probest] = ...
    svmpredict(testing_label_vector, testing_instance_matrix, model, libsvm_options)
% FUNCTION makes class label predictions on testing data together with
% model constructed from training data.
%
%   [predicted_label, accuracy, dval_probest] = ...
%       svmpredict( testing_label_vector, testing_instance_matrix, model, libsvm_options);
%
% INPUT :
%         -testing_label_vector:
%             An m by 1 vector of prediction labels. If labels of test
%             data are unknown, simply use any random values. (type must be double)
%         -testing_instance_matrix:
%             An m by n matrix of m testing instances with n features.
%             It can be dense or sparse. (type must be double)
%         -model:
%             The output of svmtrain.
%         -libsvm_options:
%             A string of testing options in the same format as that of LIBSVM.
%
% OUTPUT :
%         predicted_label   - labels predicted for testing data
%         accuracy          - scalar of prediction accuracy
%         dval_probest      - decision values or probability
%                             estimates (if '-b 1' is specified). If k is
%                             the number of classes, for decision values,
%                             each row includes results of predicting
%                             k(k-1)/2 binary-class SVMs
%
% OPTIONS :
%         b                 - decision values (0) or probability estimates
%                             for each point (1)