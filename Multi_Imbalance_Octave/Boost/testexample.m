%%%%%%%%%%%%% adaboost
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
ResultR1 = adaboostcartM1(traindata,trainlabel,testdata,20);
[kc1,lratec1,result2c1,accc1,gmeanc1,result5c1,fmeasurec1] = calculateFunc(testlabel,ResultR1);
% 
% %adaC2cartM1withoutGA
C=ones(1,length(trainlabel));
ResultR2 = adaC2cartM1(traindata,trainlabel,testdata,20,C);
[kc2,lratec2,result2c2,accc2,gmeanc2,result5c2,fmeasurec2] = calculateFunc(testlabel,ResultR2);

% %adaC2cartM1GA
C0=GAtest(traindata,trainlabel);
ResultR2g = adaC2cartM1(traindata,trainlabel,testdata,20,C0);
[kc2g,lratec2g,result2c2g,accc2g,gmeanc2g,result5c2g,fmeasurec2g] = calculateFunc(testlabel,ResultR2g);


% %SAMMEcart
ResultR3 = SAMMEcart(traindata,trainlabel,testdata,20);
[kc3,lratec3,result2c3,accc3,gmeanc3,result5c3,fmeasurec3] = calculateFunc(testlabel,ResultR3);

%adaboostcartNC
ResultR4 = adaboostcartNC(traindata,trainlabel,testdata,20,2);
[kc4,lratec4,result2c4,accc4,gmeanc4,result5c4,fmeasurec4] = calculateFunc(testlabel,ResultR4);

%PIBoostcart
ResultR5 = PIBoostcart(traindata,trainlabel,testdata,20);
[kc5,lratec5,result2c5,accc5,gmeanc5,result5c5,fmeasurec5] = calculateFunc(testlabel,ResultR5);

