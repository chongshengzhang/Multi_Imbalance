%(1) Reference (we implement the following algorithm):
%    Kang, S., Cho, S. & Kang P. (2015) Constructing a multi-class classifier using one-against-one approach
%    with different binary classifiers. Neurocomputing, 2015, Vol. 149, pp. 677-682.

%(2) This algorithm fits to numerical data.

%(3) The main idea of this algorithm:
%    a. for the multi-class imbalanced data, let nc be the number of classes.

%    b. using the One-Versus-One decomposition strategy, split the original data into nc*(nc-1)/2
%       sub-datasets, each sub-dataset only contains two classes.

%    c. for each sub-dataset, exhausitively try all the different classification algorithms,
%       at the end, pick the classification algorithm that achieves the best accuracy
%       (in terms of ACC, or G-mean, or F-measure, or AUC).

%    d. finally, each sub-dataset will have the best classification algorithm (and classification model) that
%       achieves the best accuracy on this sub-dataset.

%    e. in the prediction phase, see funcPre.m, all the nc*(nc-1)/2 classification models will
%       be used predict the labels of the test data instances, then use majority voting to make the final
%       prediction for every instance in the test data.

function [time1,time2,pre,C] = DOVO(train,testdata,testlabel,kfold)
tic;
labels = unique (train(:,end));
nc=length(labels);
flagc=1;

%for each class pair (i,j)
for i=1:nc
    for j=i+1:nc
        idi=(train(:,end)==labels(i));
        idj=(train(:,end)==labels(j));

        % Dij is the set of data instances whose class labels are either i or j
        Dij=[train(idi,:);train(idj,:)];
        
        clabels = unique (Dij(:,end)); % there will be only two label values in clabels, i and j.
        ctrainlabel=Dij(:,end);        % the corresponding labels for all the instances in the training data

        for ci=1:length(ctrainlabel)   % transform class labels from (i ,j) to (0 , 1)
            if ctrainlabel(ci)==clabels(1)
                ctrainlabel(ci)=0;
            else
                ctrainlabel(ci)=1;
            end
        end
        train1=[Dij(:,1:end-1),ctrainlabel];
        Dij=train1;

        % find Cbest which is the best classifier that corresponds to the minimum validation error.
        % Cbest is the id of the classification algorithm chosen,
        % while bestk is the specific parameter needed by the KNN classifier
        [Cbest,bestk] = bestClassifier(Dij,kfold);

        % keep the current classifier information in D
        % --Cbest is the id of the classification algorithm chosen;
        % --bestk is the specific parameter needed by the KNN classifier;
        % --clables only contains two values, which are i and j
        % Note that, here, we only keep the id of the classification algorithm, but not the model trained
        % from the training dataset. The ideal situation is to keep the model as well (this should be fixed).
        D{flagc}=Dij;
        C{flagc,1}=Cbest;
        C{flagc,2}=bestk;
        L{flagc}=clabels;
        flagc=flagc+1;
    end
end
time1=toc;
tic;

pre = funcPre(testdata,testlabel,C,D,L); %test phase
time2=toc;
%disp(Cbest);
