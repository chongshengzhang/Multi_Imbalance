% Reference:	
% Name: testDOAO.m
% 
% Authors: Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% Copyright: (c) 2018 Chongsheng Zhang <chongsheng DOT Zhang AT yahoo DOT com>
% 
% This file is a part of Multi_Imbalance software, a software package for multi-class Imbalance learning. 
% 
% Multi_Imbalance software is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
% as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
%
% Multi_Imbalance software is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program. 
% If not, see <http://www.gnu.org/licenses/>.
clear;
load_fspackage;
sname='Cardiotocography_ten_class_data_set_indx_fixed.mat';
load(sname);
kfold=5;
[a,b]=size(feature1);

dup_all_whole_mat = feature1;
dup_all_whole_labels = target;
c=floor(a*0.9);
traindata=dup_all_whole_mat(1:c,:);
trainlabel=dup_all_whole_labels(1:c,:);
testdata=dup_all_whole_mat(c+1:a,:);
testlabel=dup_all_whole_labels(c+1:a,:);
[pre,C] = DOAO([traindata,trainlabel],testdata,testlabel,kfold);
AC=testlabel-pre;
AN=find(AC==0);
ACC=size(AN,1)/length(testlabel);