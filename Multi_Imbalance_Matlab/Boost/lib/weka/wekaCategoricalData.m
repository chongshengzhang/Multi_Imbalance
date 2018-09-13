function dw = wekaCategoricalData(varargin)
%%%%%%%%%%%%%%%%%%%%%%%
%   Description:
%   DataWeka = wekaCategoricalData(X,Y,[xnames,ynames]);
% 
%   Transform a list of data points and a list of class headers
%   into a weka instances object (with categorical/nominal target).
%
%   By default everything is numerical except the target which can
%   be nominal if the number of values is limited.
%
%   Parameters:
%
%   X           -- The features on current trunk, each colum is a feature vector on all
%                  instances, and each row is a part of the instance
%   Y           -- The label of instances, in single column form: 1 2 3 4 5 ...
%   xnames      -- optional list of string for the names of the input
%               features
%   ynames      -- optional list of string for the names of the target
%               values. The classes in spider are supposed to be indexed
%               from 1 to Q (nb of classes). This string array refers to
%               the same ordering
%
%   Output:
% 
%   DataWeka (identifier 'dw')
%               -- Weka 'Instances' object where the target has been set.
%               Note that the target is nominal here.

xnames=[]; %x labels 
ynames=[]; %y labels
X = []; %the x matrix
Y = []; %the y matrix
if (length(varargin)>=2)
    X = varargin{1};
    Y = varargin{2};
end
if (length(varargin)>=3)
    xnames = varargin{3};
end
if (length(varargin)>=4)
    ynames = varargin{4};
end

%% Calculate the dimensions of the return object.
n = [size(X),size(Y,2)]; %since size(X) returns two values, this will be a 1x3 object
if (n(3)==1),
    n(3)=n(3)+1;
end;

%% compute the attribute names for the input
if (isempty(xnames))
    xnames = cell(1,n(2)); %preallocate for efficiency
    for i=1:n(2)
        xnames{i} = ['inp ' num2str(i)];
    end
end;

%% compute the list of values for the output
if (isempty(ynames))
   ynames = cell(1,n(3)); %preallocate for efficiency
   for i=1:n(3),
       ynames{i} = ['out ' num2str(i)];
   end
end

%% creates the FastVector for the class attribute
classValues = weka.core.FastVector(length(ynames));
for i = 1:length(ynames)
   classValues.addElement(ynames{i}); 
end

%% transform the output y into an index of the position of the y value
%% into ynames
ytmp = ones(size(Y,1),1);
for i = 1:size(Y,2)
    %% indexation in weka starts at zero
   ytmp(Y(:,i)==1)=i-1; 
end

%% creates the list of attributes, adds one for the target
attributes = weka.core.FastVector(n(2)+1); 
for i = 1:n(2)
    %% add numeric feature
    attributes.addElement(weka.core.Attribute(xnames{i}));
end
attributes.addElement(weka.core.Attribute('target',classValues));
    
%% creates the instances object

dw = weka.core.Instances('data',attributes,n(1));

%% set the class (target) index
%% note that indexation starts at 0 (hence the n(2) and not n(2)+1
dw.setClassIndex(n(2));

%% adds to the instances the list of input-output pairs contained in d
for i = 1:n(1)
   dw.add(weka.core.Instance(1.0, [X(i,:) ytmp(i)])); 
end