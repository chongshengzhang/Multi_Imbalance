function loadWeka(path)
%%%%%%%%%%%%%%%%%%%%%%%
%   Description:
%   The purpose of this function is to load the weka jar into the dynamic
%   classpath of MATLAB.
%
%   Parameters: 
%   path - the relative path of the weka.jar. If an empty string is passed
%   in, it will look in the default weka directory
%   %MATLABROOT%/java/jar/wek.jar.
%
%   Output:  A message will be printed to the screen
%   telling the user whether or not the attempt to load the jar resulted in
%   success.

wekajar = 'weka.jar';

%if they did not pass in a path, use the same location we've always
%used. The one used by Weka. Otherwise, give the customer what they want
%and use the path passed in.
if(isempty(path))
    wekajar = [matlabroot filesep 'java' filesep 'jar' filesep wekajar];
else
    wekajar = [pwd filesep path filesep wekajar];
end

if( ~isempty(dir(wekajar)))
    javaaddpath(wekajar);
else
    %print error that the classpath could not load
    sprintf('Unable to locate classpath. Import failed.')
end