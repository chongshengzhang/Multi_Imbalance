% This make.m is used under Mac (Linux?)

% add -largeArrayDims on 64-bit machines
% remove them on 32-bit machines

mex -O -largeArrayDims -c svm.cpp
mex -O -largeArrayDims -c svm_model_matlab.c
mex -O -largeArrayDims svmtrain.c svm.o svm_model_matlab.o
mex -O -largeArrayDims svmpredict.c svm.o svm_model_matlab.o

mex -O -largeArrayDims read_sparse.c

mex -O -largeArrayDims libsvmread.c
mex -O -largeArrayDims libsvmwrite.c