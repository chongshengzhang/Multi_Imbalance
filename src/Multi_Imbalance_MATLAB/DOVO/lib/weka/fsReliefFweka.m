function [out] =  fsReliefFweka(X,Y,param)
%%%%%%%%%%%%%%%%%%%%%%%
%   Description:
%   Using Weka's feature selection algorithm
%
%   Parameters:
%
%   X   -- The features on current trunk, each column is a feature vector on all
%           instances, and each row is a part of the instance
%   Y	-- The label of instances, in single column form: 1 2 3 4 5 ...
%
%   Output:
%
%   out -- A struct containing the field 'fList', a list of features chosen
%           by the feature selection algorithm.
%

A = X;
B = SY2MY(Y); %here, we have just converted the lists to SY2MY without making a data object.
t = weka.filters.supervised.attribute.AttributeSelection();
  
%% handle options
a.E = sprintf('weka.attributeSelection.ReliefFAttributeEval -M %i -D 1 -K %i -A 2', size(X,1), param.neighbor);
a.S = 'weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1';

tmp=wekaArgumentString({'-E',a.E});
tmp=wekaArgumentString({'-S',a.S},tmp);
t.setOptions(tmp);

%% train classifier
%get categorical data to define input format
cat = wekaCategoricalData(A,B);
t.setInputFormat(cat);

clear cat; %free memory for next categorical data (useful to free up JVM heap for rest of process).
cat = wekaCategoricalData(A,B);

%make filters off of data
newDat = weka.filters.Filter.useFilter(cat,t);
clear cat; %free memory again.

out.fList = [];

numF = newDat.numAttributes()-2;
for i = 0:numF
    str = newDat.attribute(i).name;
    str = str.toCharArray()';
    strArry = strread(str,'%s');
    out.fList(i+1) = str2num(strArry{2});
end