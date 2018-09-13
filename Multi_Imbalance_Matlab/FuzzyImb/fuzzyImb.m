%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: E. Ramentol, S. Vluymans, N. Verbiest, et al. , IFROWANN: Imbalanced Fuzzy-Rough Ordered Weighted
%            Average Nearest Neighbor Classification, IEEE Transactions on Fuzzy Systems 23 (5) (2015) 1622-1637.
%
% Note 1: We obtained the codes of IFROWANN from the authors, we greatly acknowledge their help and contributions;
% Note 2: IFROWANN was originally designed for binary imbalanced data.
%         In this work, we extend IFROWANN with the ECOC encoding strategy to handle multi-class imbalanced data.
%         This extension is implemented in the main function testfuzzyecoc().
%
% The following implementation is the core part of the IFROWANN algorithm, based on the above reference.
%
% FuzzyImb algorithm is based on the FRNN paper. For a test sample, FRNN finds in the training data,
% the most similar (closest) positive instance and the most similar negative instance, respectively.
% In case the test instance belongs to the positive class, FRNN finds the respective closest instance,
% first from the postive instances, then from the negative instances.
% Similarly, In case the test instance belongs to the negative class, FRNN also finds the respective closest instance,
% first from the instance instances, then from the positive instances.
%
% (1) For each test sample, this algorithm calculates its distancess to every instance the negative class,
%     and the distances to each instance in the positive class. see cLines 157-166
%
% (2) For the current test sample's distances with the negative class, sort the distances descendingly;
%     Similarly, for its distances with the positive class, sort the distances descendingly.
%
% (3) for the above two sorted distances (two arrays), multiply each distance with different weights;
%     There are W1...W6, 6 different weighting strategies. See lines 173-196.
%     Take W4 as an example, the last instance in the sorted array, which denotes the instance closest to the
%     test sample, will obtain a weight of 1, the second to last instance will receive a weight of 1/2, the third
%     from the last instance (in the sorted array) will receive a weight of 1/4, etc.
%     Therefore, the basic idea is that, for each test sample, its closest instance in the training data should
%     be given greatest weight, the distance exponentially decrease by 2, finally, the furthest distance in the
%     training dataset will be assigned the smallest weight. These weighted distances will be summed together,
%     for the negative and positive classes, respectively. See lines 198-206.
%
% (4) Compare the two weighted sum distances obtained above, the greater one will be chosen as the predicted class.
%     See line 208-212.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function prediction=fuzzyImb(traindata,trainclass,testdata,testclass,weightStrategy,gamma)

global aMaxs aMins;

%% // determine the positive and negative classes
tabu=tabulate(trainclass);
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
                        OWAweightspos = additive_gamma(length(lowerPos), length(lowerNeg),gamma);
                        OWAweightsneg = additive(length(lowerNeg));
                    else if strcmpi(weightStrategy,'W6') 
                            OWAweightspos = additive_gamma(length(lowerPos), length(lowerNeg),gamma);
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
                        OWAweightspos = additive_gamma(length(lowerPos), length(lowerNeg),gamma);
                        OWAweightsneg = additive(length(lowerNeg));
                    else if strcmpi(weightStrategy,'W6') 
                            OWAweightspos = additive_gamma(length(lowerPos), length(lowerNeg),gamma);
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
