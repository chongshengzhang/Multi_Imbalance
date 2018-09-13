curPath = pwd;

%% load weka jar, and common interfacing methods.
path(path, [curPath filesep 'lib']);
path(path, [curPath filesep 'lib' filesep 'weka']);
loadWeka(['lib' filesep 'weka']);
clear curPath;