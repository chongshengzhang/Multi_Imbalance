%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(1) This function funcPreEDOVO iteratively tries 8 different classifiers to build a corresponding model,
%    upon the training data (testdata,testlabel), D{i}
%(2) It then makes predictions using each specific best classifier for each binarized data (transformed using ECOC).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function allpre = funcPreEDOVO(testdata,testlabel,C,D)
[numberC,a]=size(C);
numbertest=length(testlabel);
allpre=zeros(numbertest,numberC);

for i=1:numberC
    train=D{i};
    testlabel=testdata(1:length(testlabel),end);
    testlabel(:)=0;
    testlabel(2)=1;

    %using the SVM classifier
    if C{i,1}==1
        model = svmtrain(train(:,end), train(:,1:end-1),'-t 3 -q');
        [predict_label_L, accuracy_L, dec_values_L] = svmpredict(testlabel, testdata, model);
        allpre(:,i)=predict_label_L;
    end

    %using the KNN classifier
    if C{i,1}==2
        predict_label_L=knnclassify(train(:,1:end-1), train(:,end), testdata, C{i,2},'cosine','random');
        allpre(:,i)=predict_label_L;
    end

    %using the Logistic Regression Classifier
    if C{i,1}==3
        [acc,predicted] = funLR(train(:,1:end-1), train(:,end), testdata,testlabel);
        allpre(:,i)=predicted;
    end

    %using the C4.5 Classifier
    if C{i,1}==4
        [predictionlabel,acc] = func45(train(:,1:end-1), train(:,end), testdata,testlabel);
        allpre(:,i)=predictionlabel;
    end

    %using the AdaBoost Classifier
    if C{i,1}==5
        [predictionlabel]=adafunc(train(:,1:end-1), train(:,end), testdata,testlabel);
        allpre(:,i)=predictionlabel;
    end

    %using the Random Forest Classifier
    if C{i,1}==6
        B = m5pbuild(train(:,1:end-1), train(:,end));
        predict_label = m5ppredict(B,testdata);
        predict_label_rf = predict_label;
        allpre(:,i)=predict_label_rf;
    end

    %using the CART Classifier
    if C{i,1}==7
        ft=treefit(train(:,1:end-1), train(:,end),'method','classification');
        predictionlabelcart0=treeval(ft,testdata);
        predictionlabelcart = predictionlabelcart0;
        allpre(:,i)=predictionlabelcart;
    end

    %using the MLP Classifier
    if C{i,1}==8
        [predictionlabel]=MLPfunc(train(:,1:end-1), train(:,end), testdata,testlabel);
        allpre(:,i)=predictionlabel;
    end
 end
