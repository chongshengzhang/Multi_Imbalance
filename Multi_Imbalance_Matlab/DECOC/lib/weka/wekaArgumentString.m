function arguments=wekaArgumentString(carray,appString)
%%%%%%%%%%%%%%%%%%%%%%%
%   Description:
%   This function builds a weka String array used for Argument passing
%
%   Parameters:
%
%   carray -
%      Argument has to be of the type : CELLARRAY
%      where the following format is chosen for argument passing
%      { '-A1', A1, '-A2', A2, ...}
%      The odd content must be a string which will be parsed by weka. 
%      Most of the weka argument strings are of the form " - CHARACTER".
%      All even arguments must be of type number or string. 
%
%   appString - The string used to call the application.
%
%   Output:
%
%   arguments - a list of java.lang.String's to be passed as arguments
%       to the feature selection function.

if(nargin ==1 )
    arguments=javaArray('java.lang.String',length(carray));
    
	for i=1:2:length(carray)
		arguments(i)=java.lang.String(carray{i});
	end
    
    for i=2:2:length(carray)
        if ( isnumeric(carray{i}))
            arguments(i)=java.lang.String(num2str(carray{i}));        
        elseif ( ischar(carray{i}))
            arguments(i)=java.lang.String(carray{i});            
        end
    end
    
else
    
    arguments=javaArray('java.lang.String',length(carray)+length(appString));
    for i=1:length(appString)
        arguments(i)=appString(i);
    end
    s=wekaArgumentString(carray);
    for i=1:length(s)
        arguments(i+length(appString))=s(i);
    end
end