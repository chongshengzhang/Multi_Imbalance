function classifications = knnclassify(train_points, train_labels, test_points, k);

%-------------------------------------------
% K nearest neighbour (KNN) classification
% code by Jaakko Peltonen 2008
%-------------------------------------------
% Classifies test points based on majority vote of their k nearest
% neighbours in train points. 'Nearest' is determined from squared 
% Euclidean distance. In case the majority voting results in a tie,
% gives equal portions to each class in the tie.
%
% Inputs:
%---------
% train_points: matrix, the i:th row has the features of the i:th 
%               training point.
%
% train_labels: two possible formats. 
%   Format 1: a column vector where the i:th element is an integer 
%             value indicating the label of the i:th training
%             point. The labels should start from zero.
%
%   Format 2: a matrix where the i:th row has the class memberships
%             of the i:th training point. Each class membership is
%             a value from 0 to 1, where 1 means the point fully 
%             belongs to that class.
%
% test_points: feature matrix for test points. Same format as
%    train_points. You can give an empty matrix: then the method
%    computes leave-one-out classification error based only on
%    the training points.
%
% Outputs:
%---------
% classifications: matrix, the i:th row has the predicted class
%             memberships of the i:th test point. Each class
%             membership if a value from 0 to 1, where 1 means
%             the test point is predicted to fully belong to
%             that class.
%


nDim = size(test_points,2);
nTrainPoints = size(train_points,1);
nClasses = numel(unique(train_labels));

% if the training labels were provided in format 1,
% convert them to format 2.

if size(train_labels,2)==1,
  nClasses = numel(unique(train_labels));
  train_labels2 = zeros(nTrainPoints,nClasses);
  for i=1:nTrainPoints,
    train_labels2(i,train_labels(i)+2) = 1;
  end;
  train_labels = train_labels2;
end;
nClasses = size(train_labels,2);

% if test_points is empty, perform leave-one-out classification
if isempty(test_points),
  test_points = train_points;
  leave_one_out = 1;
else
  leave_one_out = 0;
end;
nTestPoints = size(test_points,1);


% 
% Perform the KNN classification. For leave-one-out classification,
% this code assumes that k < nTrainPoints.
%
classifications = zeros(nTestPoints, nClasses);
for i=1:nTestPoints,
  % find squared Euclidean distances to all training points
  difference = train_points(:,1:nDim) - repmat(test_points(i,:),[nTrainPoints 1]);
  distances = sum(difference.^2,2);

  % in leave-one-out classification, make sure the point being
  % classified is not chosen among the k neighbors.
  if leave_one_out == 1,
    distances(i) = inf;
  end;  
  
  % collect the 'votes' of the k closest points
  [sorted_distances, indices] = sort(distances);
  classamounts = zeros(1, nClasses);
  for j=1:k,
    classamounts = classamounts + train_labels(indices(j),:);
  end;
  
  % choose the class by majority vote
  indices = find(classamounts == max(classamounts));
  if (length(indices) == 1),
    % there is a single winner
    classifications(i,indices(1)) = 1;
  else
    % there was a tie between two or more classes
    classifications(i,indices) = 1/length(indices);
  end;
end;

classifications = classifications(:,3);