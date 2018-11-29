% Reference:	
% Name: fuzzyImb.m
% 
% Purpose: Imbalanced Fuzzy-Rough Ordered Weighted Average Nearest Neighbor Classification
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
function prediction=fuzzyImb(traindata,trainclass,testdata,testclass,weightStrategy,gamma)

global aMaxs aMins;

%% // determine the positive and negative classes
tabu=tabulate(trainclass,[0,1,2]);
if tabu(1,2)>tabu(2,2)
    posID=tabu(2,1);
    negID=tabu(1,1);
else
    posID=tabu(1,1);
    negID=tabu(2,1);
end

%% // compute attribute ranges
%% // for each numeric (float) attribute, derive the minimum and maximum values, and keep them in aMins and aMaxs
%% // for a categorical attribute, they do not need to compute the minimum and maximum values
[atrain,btrain]=size(traindata);
aMins = zeros(1,btrain);
aMaxs = zeros(1,btrain);
for a = 1:btrain
    %%// if not a NOMINAL Attribute (but a numeric attribute)
    if 1
        min0 =  realmax;
        max0 = - realmax;
        %% // derive the minimum and maximum values for each attribute
        for x = 1:atrain
            min0 = min(min0,traindata(x,a));
            max0 = max(max0,traindata(x,a));
        end
        aMins(1,a) = min0;
        aMaxs(1,a) = max0;
    end
end


%% // working on training data
trainRealClass = zeros(length(trainclass),1);
trainPrediction = zeros(length(trainclass),1);
         
for i = 1:length(trainclass)
    %% // true class
    trainRealClass(i,1) = trainclass(i);
    % %     //////////////////////////////////////////////
    % %     // predicted class with the IFROWANN method //
    % %     //////////////////////////////////////////////
    nlowerPos=1;
    nlowerNeg=1;
    for x = 1:length(trainclass)
        if trainclass(x) == posID
            lowerNeg(nlowerNeg)=1.0 - similarity(traindata(x,:), traindata(i,:));
            nlowerNeg=nlowerNeg+1;
        else if trainclass(x) == negID
            lowerPos(nlowerPos)=1.0 - similarity(traindata(x,:), traindata(i,:));
            nlowerPos=nlowerPos+1;
            end
        end
    end
    
    lowerPos=sort(lowerPos,'descend');
    lowerNeg=sort(lowerNeg,'descend');
    
    %% // obtain the OWA weights
    if strcmpi(weightStrategy,'W1')
        OWAweightspos = additive(length(lowerPos));
        OWAweightsneg = additive(length(lowerNeg));
    else if strcmpi(weightStrategy,'W2')
            OWAweightspos = additive(length(lowerPos));
            OWAweightsneg = exponential(length(lowerNeg));
        else if strcmpi(weightStrategy,'W3')
                OWAweightspos = exponential(length(lowerPos));
                OWAweightsneg = additive(length(lowerNeg));
            else if strcmpi(weightStrategy,'W4') 
                    OWAweightspos = exponential(length(lowerPos));
                    OWAweightsneg = exponential(length(lowerNeg));
                else if strcmpi(weightStrategy,'W5')
                        OWAweightspos = additiveGamma(length(lowerPos), length(lowerNeg),gamma);
                        OWAweightsneg = additive(length(lowerNeg));
                    else if strcmpi(weightStrategy,'W6') 
                            OWAweightspos = additiveGamma(length(lowerPos), length(lowerNeg),gamma);
                            OWAweightsneg = exponential(length(lowerNeg));
                        end
                    end
                end
            end
        end
    end
    
    sumPos = 0.0;
    for el = 1:length(lowerPos)
        sumPos = sumPos + OWAweightspos(el) * lowerPos(el);
    end
    
    sumNeg = 0.0;
    for el = 1:length(lowerNeg)
        sumNeg = sumNeg + OWAweightsneg(el) * lowerNeg(el);
    end
    
    if sumPos >= sumNeg
        trainPrediction(i,1) = posID;
    else
        trainPrediction(i,1) = negID;
    end
end

%% // Working on test data
[testlength,testa]=size(testdata);
realClass = zeros(testlength,1);
prediction = zeros(testlength,1);

for i = 1:length(realClass)
    %%  true class
    realClass(i,1) = testclass(i);
    % %     //////////////////////////////////////////////
    % %     // predicted class with the IFROWANN method //
    % %     //////////////////////////////////////////////
    nlowerPos=1;
    nlowerNeg=1;
    for x = 1:length(trainclass)
        if trainclass(x) == posID
            lowerNeg(nlowerNeg)=1.0 - similarity(traindata(x,:), testdata(i,:));
            nlowerNeg=nlowerNeg+1;
        else if trainclass(x) == negID
            lowerPos(nlowerPos)=1.0 - similarity(traindata(x,:), testdata(i,:));
            nlowerPos=nlowerPos+1;
            end
        end
    end
    
    lowerPos=sort(lowerPos,'descend');
    lowerNeg=sort(lowerNeg,'descend');

    %% // obtain the OWA weights
   
    if strcmpi(weightStrategy,'W1')
        OWAweightspos = additive(length(lowerPos));
        OWAweightsneg = additive(length(lowerNeg));
    else if strcmpi(weightStrategy,'W2')
            OWAweightspos = additive(length(lowerPos));
            OWAweightsneg = exponential(length(lowerNeg));
        else if strcmpi(weightStrategy,'W3')
                OWAweightspos = exponential(length(lowerPos));
                OWAweightsneg = additive(length(lowerNeg));
            else if strcmpi(weightStrategy,'W4') 
                    OWAweightspos = exponential(length(lowerPos));
                    OWAweightsneg = exponential(length(lowerNeg));
                else if strcmpi(weightStrategy,'W5')
                        OWAweightspos = additiveGamma(length(lowerPos), length(lowerNeg),gamma);
                        OWAweightsneg = additive(length(lowerNeg));
                    else if strcmpi(weightStrategy,'W6') 
                            OWAweightspos = additiveGamma(length(lowerPos), length(lowerNeg),gamma);
                            OWAweightsneg = exponential(length(lowerNeg));
                        end
                    end
                end
            end
        end
    end
    
    sumPos = 0.0;
    for el = 1:length(lowerPos)
        sumPos = sumPos + OWAweightspos(el) * lowerPos(el);
    end
    
    sumNeg = 0.0;
    for el = 1:length(lowerNeg)
        sumNeg = sumNeg + OWAweightsneg(el) * lowerNeg(el);
    end
    
    if sumPos >= sumNeg
        prediction(i,1) = posID;
    else
        prediction(i,1) = negID;
    end
end
