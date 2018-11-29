function wekaOBJ = matlab2weka2(name, featureNames, data,classi)
%__________________________________________________________________________
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % MATLB2WEKA2 % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % %  % % Leonardo Onofri 2011 % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%                                                                         
%--------------------------------------------------------------------------
%                              INPUT ARGUMENTS                            
%--------------------------------------------------------------------------
%__ name: a string, naming the data/relation                               
%__ featureNames: A cell array of d+1 strings, where d is the number of
%                 of features, naming each feature. The last element must
%                 be the string 'class';
%__ data: an n-by-d matrix of class double with n, d-featured samples /(feature numeriche, matrice ogni riga un campione). 
%__ classi: an n-by-1 cell array containing the class of each sample (vettore colonna, in celle).
%
%--------------------------------------------------------------------------
%                             OUTPUT ARGUMENTS                            
%--------------------------------------------------------------------------
%__ wekaOBJ: Returns a java object of type weka.core.Instances          
%                                                                         
%--------------------------------------------------------------------------
%                                DESCRIPTION                              
%--------------------------------------------------------------------------
% Convert matlab data to a weka java Instances object for use by weka
% classes. Originally written by Matthew Dunham. Modified by Leonardo
% Onofri for numerical attributes.
%                                                                         
%--------------------------------------------------------------------------
%                                   USAGE                                 
%--------------------------------------------------------------------------
% wekaOBJ = matlab2weka2('dataset', {'feat1','feat2','feat3','class'},...
%           rand(4,3),{'classe1';'classe1';'classe2';'classe2'});         
%__________________________________________________________________________


if(~wekaPathCheck),wekaOBJ = []; return,end

targetIndex=size(data,2)+1;

import weka.core.*;
vec = FastVector();

for i=1:numel(featureNames)-1
    vec.addElement(Attribute(featureNames{i}));
end

attvals = unique(classi);
values = FastVector();
for j=1:numel(attvals)
    values.addElement(attvals{j});
end
vec.addElement(Attribute(featureNames{end},values));

wekaOBJ = Instances(name,vec,size(data,1));
for i=1:size(data,1)
    inst=Instance(1,[data(i,:) 1]);
    inst.setDataset(wekaOBJ);
    inst.setValue(numel(featureNames)-1,classi{i});
    wekaOBJ.add(inst);
end

wekaOBJ.setClassIndex(targetIndex-1);
end