% Reference:	
% Name: testExample.m
% 
% Purpose: an example of method tested
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
%   load('biodata.mat');
%  load('walldata.mat')
%   load('autodata.mat');
  load('6alrawdata.mat');


% cart
    
ft = classregtree(traindata,trainlabel,'method','classification');   
prec=eval(ft,testdata);
prec=cellfun(@str2num, prec);
[kc,lratec,result2c,accc,gmeanc,result5c,fmeasurec] = calculateFunc(testlabel,prec);



% % % % % % % % %adaboostcart
% % % % % % % ResultR = adaboostcart(traindata,trainlabel,testdata,3);
% % % % % % % [kc0,lratec0,result2c0,accc0,gmeanc0,result5c0,fmeasurec0] = calculateFunc(testlabel,ResultR);
% 




% %adaboostcartM1
ResultR1 = adaBoostCartM1(traindata,trainlabel,testdata,20);
[kc1,lratec1,result2c1,accc1,gmeanc1,result5c1,fmeasurec1] = calculateFunc(testlabel,ResultR1);
% 
% %adaC2cartM1withoutGA
C=ones(1,length(trainlabel));
ResultR2 = adaC2CartM1(traindata,trainlabel,testdata,20,C);
[kc2,lratec2,result2c2,accc2,gmeanc2,result5c2,fmeasurec2] = calculateFunc(testlabel,ResultR2);

% %adaC2cartM1GA
C0=testGA(traindata,trainlabel);
ResultR2g = adaC2CartM1(traindata,trainlabel,testdata,20,C0);
[kc2g,lratec2g,result2c2g,accc2g,gmeanc2g,result5c2g,fmeasurec2g] = calculateFunc(testlabel,ResultR2g);


% %SAMMEcart
ResultR3 = sammeCart(traindata,trainlabel,testdata,20);
[kc3,lratec3,result2c3,accc3,gmeanc3,result5c3,fmeasurec3] = calculateFunc(testlabel,ResultR3);

%adaboostcartNC
ResultR4 = adaBoostCartNC(traindata,trainlabel,testdata,20,2);
[kc4,lratec4,result2c4,accc4,gmeanc4,result5c4,fmeasurec4] = calculateFunc(testlabel,ResultR4);

%PIBoostcart
ResultR5 = PIBoostCart(traindata,trainlabel,testdata,20);
[kc5,lratec5,result2c5,accc5,gmeanc5,result5c5,fmeasurec5] = calculateFunc(testlabel,ResultR5);

