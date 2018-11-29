# Multi_Imbalance
**Implementations of 18 major (state-of-the-art) Multi-class Imbalance learning (Imbalance Classification) algorithms.**

###### Copyright: Jingjun Bi and Chongsheng Zhang (The Big Data Research Center, Henan University).

###### The two contributors of this open software for multi-class imbalance learning are  Ms. Jingjun Bi and Prof. Chongsheng Zhang (chongsheng.zhang@yahoo.com). If you have any problems, please do not hesitate to send us an email. 

# (1) Installation

1. Software Environments: Please first install and use the Matlab or OCTAVE softwares. 

2. Please download the software to your local disk, then add the folder 'Multi_Imbalance' and all subfolders to the path of Matlab or OCTAVE.

*The users can run these codes in the Windows Operating Systems or Mac OSx.*

*If you run the codes with Mac OSx, there will be issues with the LibSVM software package we have included. In this case, users just ignore (stop using) LibSVM.*


# (2) Software Contents

The software has 9 major folders which are described below.

1. The folder 'Boost' contains a class of 5 Boosting methods: AdaBoost.M1, AdaC2.M1, SAMME, AdaBoost.NC, PIBoost).

2. The folder 'DECOC' contains algorithm DECOC.

3. The folder 'DOVO' contains the algorithm DOVO.

4. The folder 'FuzzyImb' contains algorithm FuzzyImb+ECOC.

5. The folder 'HDDT' contains a class of 3 methods: MC-HDDT, HDDT+ECOC, HDDT+OVA.

6. The folder 'imECOC' contains a class of 3 methods: imECOC+sparse, imECOC+dense, imECOC+OVA.

7. The folder 'MultiIM' contains a class of 4 methods: Multi-IM+OVO, Multi-IM+A&O, Multi-IM+OAHO, Multi-IM+OVA.  

8. The folder 'data' is the experimental datasets, and each experiment is carried out using 5-fold cross validation. 

9. The folder 'results' are the experimental results, '_c.mat' is time consuming, and '_p.mat' is prediction labels.

 <br />
 
*In total,  18 algorithms for multi-class imbalance learning. Note that we also include the 19th algorithm CART as the baseline method, but CART is not towards imbalance learning. The file 'testall.m' is an example of all the methods tested, a total of 19 methods including the above 18 algorithms and the base classifier CART.*


# (3) Software Usage
There are 7 classes (categories) of algorithms for multi-class imbalance learning, each class consisting of one or more algorithms. 

In total, there are 18 major algorithms for multi-class imbalance learning.

**The detailed documentation of Multi_Imbalance is available at  https://github.com/chongshengzhang/Multi_Imbalance/blob/master/doc/User_manual_Matlab.pdf and https://github.com/chongshengzhang/Multi_Imbalance/blob/master/doc/User_manual_Octave.pdf. The documentation describes the principles and usage of algorithms in Multi_Imbalance. We recommend users to refer to the above documentaion.**

In the following, we also provide a brief user manual for this software.

**The inputs and outputs for these 18 algorithms are the same.**

<br />

**Input**:
**data(d).train, data(d).trainlabel, data(d).test**

*data(d).train and data(d).trainlabel are the training data matrix and the correponding lables, data(d).test is testing data matrix.*

<br />

**Output**:
**Cost(d).NAME, Pre(d).NAME**

*Pre(d).NAME is the prediction labels, and Cost(d).NAME is the time consumption.*

 <br />

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# 1. Variants of AdaBoost for Multi-class Imbalance Learning 


  *These 5 algorithms are under the folder "Boost", all of them are variants of Boost/AdaBoost.  AdaBoost (Adaptive Boosting) is a binary classification algorithm proposed by Freund and Schapire that integrates multiple weak classifiers to build a stronger classifier. AdaBoost only supports binary data in the beginning, but it was later extended to multi-class scenarios. AdaBoost.M1 and SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss function) have extended AdaBoost in both the update of samples’ weights and the classifier combination strategy. The main difference between them is the method  for updating the weights of the samples.*

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Principles

**(1) AdaBoost.M1.** 
   
   The main steps of AdaBoost.M1 are as follows:

   Step 1: Initialize the weight Vector with uniform distribution

   Step 2: for t=1 to Max_Iter do

   Step 3:    Fit a classifier nb to the training data using weights

   Step 4:    Compute weighted error

   Step 5:    Compute AlphaT=0.5*log((CorrectRate+eps)/(errorRate+eps))

   Step 6:    Update weights

   Step 7:    Re-normalize weight

   Step 8: end for

   Step 9: Output Final Classifier

   *Reference for AdaBoost.M1: Freund, Y. & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, August 1997, 55(1).*


 **(2) SAMME.**
 
   The main procedure (steps) of the SAMME algorithm are as follows:
	
   Step 1: Initialize the weight Vector with uniform distribution

   Step 2: for t=1 to Max_Iter do

   Step 3:    Fit a classifier nb to the training data using weights

   Step 4:    Compute weighted error: errorRate=sum(weight(find(predicted~=trainlabel)));

   Step 5:    Compute AlphaT=log((1-errorRate)/(errorRate+eps))+log(length(labels)-1)

   Step 6:    Update weights weight(i)=weight(i)* exp( AlphaT(t));(trainlabel(i)~=predicted(i))

   Step 7:    Re-normalize weight

   Step 8: end for

   Step 9: Output Final Classifier

   *Reference for SAMME:  Zhu, J., Zou, H., Rosset, S., et al. (2006). Multi-class AdaBoost. Statistics & Its Interface, 2006, 2(3), 349-360.*

 **(3) AdaC2.M1 (adaC2cartM1).**
 
 *AdaC2.M1 derives the best cost setting through the genetic algorithm (GA) method, then takes this cost setting into consideration in the subsequent boosting. Genetic algorithm proposed by Holland is based on natural selection and genetics of random search technology. GA can achieve excellent performance in finding the best parameters.*

 *Reference for AdaC2.M1: Sun, Y., Kamel, M. S. & Wang, Y. (2006). Boosting for learning multiple classes with imbalanced class
  distribution. Proceedings of the 6th International Conference on Data Mining, 2006 (PP. 592-602).*
  
**(4) AdaBoost.NC.**

Since GA is very time consuming, in the above reference, the authors propose AdaBoost.NC which deprecates the GA algorithm, but emphasizes ensemble diversity during training, and exploits its good generalization performance to facilitate class imbalance learning.

*Reference for AdaBoost.NC: Wang, S., Chen, H. & Yao, X. Negative correlation learning for classification ensembles. Proc. Int. Joint
  Conf. Neural Netw., 2010 (PP. 2893-2900).*

**(5) PIBoost.**

PIBoost combines binary weak-learners to separate groups of classes, and uses a margin-based exponential loss function to classify multi-class imbalanced data.

*Reference for PIBoost: Fernndez, B. A. & Baumela. L. (2014). Multi-class boosting with asymmetric binary weak-learners. Pattern Recognition, 2014, 47(5), PP. 2080-2090.*
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Usage
## 1.1 AdaBoost.M1
    % The main function is: adaboostcartM1()
    % The principles and procedure (and the rational) of this algorithm are explained/given in adaboostcartM1.m.
			
    % Usage for adaboostcartM1:
		
    function runAdaBoostM1
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);
      
    %AdaBoost.M1
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
    
    [Cost(d).adaboostcartM1tr,Cost(d).adaboostcartM1te,Pre(d).adaboostcartM1] = adaBoostCartM1(data(d).train,data(d).trainlabel,data(d).test,20);
        
        end
        
        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 1.2 SAMME
    % The main function is: SAMMEcart()
    % The principles and procedure (and the rational) of this algorithm are explained/given in SAMMEcart.m.
			
    % Usage for adaboostcartM1:
		
    function runSAMME
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %SAMME
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).SAMMEcarttr,Cost(d).SAMMEcartte,Pre(d).SAMMEcart] = sammeCart(data(d).train,data(d).trainlabel,data(d).test,20);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 1.3 AdaC2.M1
    % The main function is: adaC2cartM1()
    % The principles and procedure (and the rational) of this algorithm are explained/given in adaC2cartM1.m.
			
    % Usage for adaC2cartM1:
		
    function runAdaC2M1
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %AdaC2.M1
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter,C
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
            tic;
            C0=GAtest(data(d).train,data(d).trainlabel);
            Cost(d).GA=toc;
            Indx(d).GA=C0;
        
            [Cost(d).adaC2cartM1GAtr,Cost(d).adaC2cartM1GAte,Pre(d).adaC2cartM1GA] = adaC2CartM1(data(d).train,data(d).trainlabel,data(d).test,20,C0);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 1.4 AdaBoost.NC
    % The main function is: adaboostcartNC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in adaboostcartNC.m.
			
    % Usage for adaboostcartNC:
		
    function runAdaBoostNC
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %AdaBoost.NC
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter,lama
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
    
            [Cost(d).adaboostcartNCtr,Cost(d).adaboostcartNCte,Pre(d).adaboostcartNC] = adaBoostCartNC(data(d).train,data(d).trainlabel,data(d).test,20,2);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 1.5 PIBoost
    % The main function is: PIBoostcart()
    % The principles and procedure (and the rational) of this algorithm are explained/given in PIBoostcart.m.
			
    % Usage for PIBoostcart:
		
    function runPIBoost
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %PIBoost
    %input: traindata,trainlabel,testdata,testlabel,Max_Iter
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).PIBoostcarttr,Cost(d).PIBoostcartte,Pre(d).PIBoostcart] = PIBoostCart(data(d).train,data(d).trainlabel,data(d).test,20);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;
    end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# 2. DECOC 
*This algorithm is our proposed ensemble approach for imbalance learning, DOVO+imECOC.*

*Reference of DECOC: Jingjun Bi, Chongsheng Zhang*. (2018). An Empirical Comparison on State-of-the-art Multi-class Imbalance  Learning Algorithms and A New Diversified Ensemble Learning Scheme.  Knowledge-based Systems, 2018.*


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Principles

Our DECOC algorithm contains the following steps:

Step 1: DECOC first uses ECOC strategy to tranform the multi-class data into multiple binary data, then finds the best classifier for each specific binaried data, which will be kept by ft.

Step 2: Using funcwEDOVO, DECOC builds the best classifier for each binarized data (by ECOC) and the predictions are in allpre.

Step 3: With funcpretestEDOVO, DECOC makes the predictions on the test data, using the imECOC algorithm. 

# Usage
## 2.1 DECOC
    % The main function is: DECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in DECOC.m.
	
    % Usuage of the DECOC method: 
		
   function runDECOC
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

        %DECOC
        %input: traindata,trainlabel,testdata,testlabel,type,withw
        %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
    
            [Cost(d).imECOCDOVOs1tr,Cost(d).imECOCDOVOs1te,Pre(d).imECOCDOVOs1] = DECOC(data(d).train,data(d).trainlabel,data(d).test, 'sparse',1);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






# 3. DOVO 
     (This algorithm is an ensemble based approach for imbalance learning)

*Reference (we implement the following algorithm): Kang, S., Cho, S. & Kang P. (2015) Constructing a multi-class classifier using one-against-one approach with different binary classifiers. Neurocomputing, 2015, Vol. 149, pp. 677-682.* 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Principles

The main idea of this algorithm:

Step 1. for the multi-class imbalanced data, let nc be the number of classes.

Step 2. using the One-Versus-One decomposition strategy, split the original data into nc*(nc-1)/2 sub-datasets, each sub-dataset only contains two classes.

Step 3. for each sub-dataset, exhausitively try all the different classification algorithms, at the end, pick the classification algorithm that achieves the best accuracy (in terms of ACC, or G-mean, or F-measure, or AUC).

Step 4.  finally, each sub-dataset will have the best classification algorithm (and classification model) that achieves the best accuracy on this sub-dataset.

Step 5.  in the prediction phase, see funcPre.m, all the nc*(nc-1)/2 classification models will be used predict the labels of the test data instances, then use majority voting to make the final prediction for every instance in the test data. 

# Usage
## 3.1 DOVO
    % The main function is: DOVO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in DOVO.m.
	
    % Usuage of the DOVO method: 
    function runDOVO
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %DOVO
    %input: traindata,trainlabel,testdata,testlabel,kfold
    %output: trainCostTime,predictCostTime,predictResult,bestChosen
        for d=1:5

            [Cost(d).DOAOtr,Cost(d).DOAOte,Pre(d).DOAO,Indx(d).C] = DOVO([data(d).train,data(d).trainlabel],data(d).test,data(d).testlabel,5);

        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




# 4. fuzzyImbECOC 
  *This algorithm is a combination of FuzzyImb and ECOC, i.e., FuzzyImb+ECOC.*

*Reference: E. Ramentol, S. Vluymans, N. Verbiest, et al. , IFROWANN: Imbalanced Fuzzy-Rough Ordered Weighted Average Nearest Neighbor Classification, IEEE Transactions on Fuzzy Systems 23 (5) (2015) 1622-1637.*

*We obtain the codes of IFROWANN (fuzzyImb) from the authors, we greatly acknowledge their help and contributions. It was originally designed for binary imbalanced data. In this work, we extend IFROWANN with the ECOC encoding strategy to handle multi-class imbalanced data.*


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Principles

The main procedure of fuzzyImbECOC:

Step 1. Using funECOC(), fuzzyImbECOC first generates the ECOC matrix (with each codeword for a specific class), each class will be represented by an array of codes such as 1 1 -1 -1 1 -1.

Step 2. It next extracts the instances (and the corresponding labels) of each original class, keep in train{i};

Step 3. the ECOC matrix for all the classes is an nc*number1 matrix, each row represents the codeword of one class.

   a)  for each column of the ECOC matrix, first retrieve the corresponding bit value of each class, then assign this bit value as the label for all the instances of the current class.  This is for handling the multi-class data.

   b)  for each two-class data, use fuzzyImb to train the binary classifier (see above reference). At the end, we will train a few binary classifiers.

   c)  for each test instance from testdata, use all the classifiers obtained from b) to make predictions; their predicitions will be  combined as an array, then use the ECOC decoding method to find the nearest ECOC codeword, then the  corresponding class label.

# Usage
## 4.1 fuzzyImbECOC
    % The main function is: fuzzyImbECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in fuzzyImbECOC.m.
			
    % Usage for fuzzyImbECOC (FuzzyImb+ECOC)
    function runFuzzyImbECOC
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %FuzzyImb+ECOC
    %input: traindata,trainlabel,testdata,testlabel,weightStrategy,gamma
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
            tic;
            [Pre(d).fuzzyw6] = fuzzyImbECOC(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel, 'w6',0.1);
            Cost(d).fuzzyw6=toc;
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





# 5. HDDT 
  *a class of 3 algorithms in total, all of which are variants of HDDT.*

*Reference： Hoens, T. R., Qian, Q., Chawla, N. V., et al. (2012). Building decision trees for the multi-class imbalance problem.    Advances in Knowledge Discovery and Data Mining. Springer Berlin Heidelberg, 2012 (PP. 122-134).*


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Principles
  
**(1)HDDTova**

*HDDTova is HDDT plus the decomposition technique OVA for multi-class imbalanced data. It is our own extension of HDDT to multi-class imbalanced data. HDDTova builds numberc of binary HDDT classifiers by combining the OVA strategy and HDDT, then combines the outputs of different binary HDDT classifiers generated using the OVA strategy. Here, the decoding strategy for OVA is the same as the imECOC decoding. It must be noted that the decoding strategy for testHDDTecoc and testHDDTova are identical, for fair comparisons between them.*

**(2)HDDTecoc**

*HDDTecoc is HDDT plus the decomposition technique ECOC for multi-class imbalanced data. It builds number1 of binary HDDT classifiers by combining the ECOC strategy and HDDT. Then combines the outputs of different binary HDDT classifiers generated using the ECOC strategy.*

**(3)MCHDDT**

*MCHDDT, the Multi-Class HDDT method, successively takes one or a pair of classes as the positive class and the rest as negative class, when calculating the Hellinger distance for each feature. It next selects the maximum Hellinger value for this feature. Finally, it obtains the maximum Hellinger value for each feature, then the feature with maximum Hellinger distance will be used to split the node. Then, after determining the best split feature, it recursively build the child trees.*

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Usage
## 5.1 HDDTova
    % The main function is: HDDTova()
    % The principles and procedure (and the rational) of this algorithm are explained/given in HDDTova.m.
			
    % Usage for HDDTova:
		
    function runHDDTOVA
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %HDDT+OVA
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).HDDTovatr,Cost(d).HDDTovate,Pre(d).HDDTova] = HDDTOVA(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);
            
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

## 5.2 HDDTecoc
    % The main function is: HDDTecoc()
    % The principles and procedure (and the rational) of this algorithm are explained/given in HDDTecoc.m.
			
    % Usage for HDDTecoc:
		
    function runHDDTECOC
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %HDDT+ECOC
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5

            [Cost(d).HDDTecoctr,Cost(d).HDDTecocte,Pre(d).HDDTecoc] = HDDTECOC(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);

        end
    
        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end

		

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

## 5.3 MCHDDT
    % The main function is: MCHDDT()
    % The principles and procedure (and the rational) of this algorithm are explained/given in MCHDDT.m.
			
    % Usage for MCHDDT:
		
    function runMCHDDT
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %MC-HDDT
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
    
            [Cost(d).MCHDDTtr,Cost(d).MCHDDTte,Pre(d).MCHDDT] = MCHDDT(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# 6. imECOC 
  *a class of 3 algorithms in total, all of which are variants of imECOC.*

*Reference: Liu, X. Y., Li, Q. Q. & Zhou Z H. (2013). Learning imbalanced multi-class data with optimal dichotomy weights. IEEE 13th International Conference on Data Mining (IEEE ICDM), 2013 (PP.  478-487).*


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Principles

The imECOC algorithm contains the following steps:

Step 1. in each binary classifier, it simultaneously considers the between-class and the within-class imbalance; see function funclassifier();

Step 2. in the training/prediction phase, it assigns different weights to different binary classifiers; see function funcw();

Step 3. in the prediction phase, it decodes it with weighted distance to obtain the optimal weight of the classifier by minimizing the weighted loss;  see function funcpre().

*In our implementations, Sparse, Dense, OVA are the 3 different coding methods for ECOC.*


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Usage
## 6.1 imECOC+sparse
    % The main function is: imECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in imECOC.m.
			
    % Usage for imECOC+sparse:
		
    function runImECOCsparse
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %imECOC+sparse
    %input: traindata,trainlabel,testdata,testlabel,type,withw
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).imECOCs1tr,Cost(d).imECOCs1te,Pre(d).imECOCs1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'sparse',1);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

## 6.2 imECOC+OVA
    % The main function is: imECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in imECOC.m.
			
    % Usage for imECOC+OVA:
		
    function runImECOCOVA
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %imECOC+OVA
    %input: traindata,trainlabel,testdata,testlabel,type,withw
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).imECOCo1tr,Cost(d).imECOCo1te,Pre(d).imECOCo1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'OVA',1);
            
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
		
## 6.3 imECOC+dense
    % The main function is: imECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in imECOC.m.
			
    % Usage for imECOC+dense:
            
    function runImECOCdense
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %imECOC+dense
    %input: traindata,trainlabel,testdata,testlabel,type,withw
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).imECOCd1tr,Cost(d).imECOCd1te,Pre(d).imECOCd1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'dense',1);
        
        end
    
        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end

		
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# 7. MultiIM 
  *a class of 4 algorithms in total, all of which are variants of Multi-IM.*

*Reference: Ghanem, A. S., Venkatesh, S. & West, G. (2010). Multi-class pattern classification in imbalanced data. International Conference on Pattern Recognition (ICPR), 2010 (PP. 2881-2884).*


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Principles

*PRMs-IM is a classification algorithm (originally designed) for binary imbalanced data. Let m be the ratio between the number of majority samples and that of the minority samples. PRMs-IM randomly divides the majority samples into m parts, next combines each part with all the minority instances, then trains a corresponding binary classifier.  In the prediction phase, it uses weighted voting to ensemble the outputs of the m classifiers and makes the final prediction.*

*Multi-IM algorithm combines A&O and PRMs-IM, where PRMs-IM is adopted to train the classifier for A&O. Besides A&O, in our work, we also combine the OVA, OVO and OAHO decomposition methods with PRMs-IM to further investigate the performance of PRMs-IM for multi-class imbalance learning.*
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Usage
## 7.1 Muti-IM+OVA
    % The main function is: classOVA()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classOVA.m.
			
    % Usage for Muti-IM+OVA:
		
    function runMultiImOVA
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %Multi-IM+OVA
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).classOVAtr,Cost(d).classOVAte,Pre(d).classOVA] = classOVA(data(d).train,data(d).trainlabel,data(d).test);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

## 7.2 Muti-IM+OVO
    % The main function is: classOAO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classOVO.m.
		
    % Usage for Muti-IM+OVO:		
    
    function runMultiImOVO
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %Multi-IM+OVO
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).classOAOtr,Cost(d).classOAOte,Pre(d).classOAO] = classOAO([data(d).train,data(d).trainlabel],data(d).test);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 7.3 Muti-IM+OAHO
    % The main function is: classOAHO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classOAHO.m.
		
    % Usage for Muti-IM+OAHO:		
        
    function runMultiImOAHO
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %Multi-IM+OAHO
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5

            [Cost(d).classOAHOtr,Cost(d).classOAHOte,Pre(d).classOAHO] = classOAHO([data(d).train,data(d).trainlabel],data(d).test);
        
        end

        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 7.4 Muti-IM+A&O
    % The main function is: classAandO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classAandO.m
		
    % Usage for Muti-IM+A&O:	    
 
    function runMultiImAO
    javaaddpath('weka.jar');

    p = genpath(pwd);
    addpath(p, '-begin');
    % record = 'testall.txt';
    % save record record

    dataset_list = {'Wine_data_set_indx_fixed'};

    for p = 1:length(dataset_list)%1:numel(dataset_list)
        load(['data\', dataset_list{p},'.mat']);
        disp([dataset_list{p}, ' - numero dataset: ',num2str(p), ]);

    %Multi-IM+A&O
    %input: traindata,trainlabel,testdata,testlabel,
    %output: trainCostTime,predictCostTime,predictResult
        for d=1:5
        
            [Cost(d).classAandOtr,Cost(d).classAandOte,Pre(d).classAandO] = classAandO(data(d).train,data(d).trainlabel,data(d).test);
        
        end
    
        save (['results/', dataset_list{p},'_',  'p', '.mat'], 'Pre');
        save (['results/', dataset_list{p},'_', 'c', '.mat'],  'Cost');
        
        clear Cost Pre Indx;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This is the END. Thank you so much.


Prof.Chongsheng Zhang (chongsheng.zhang@yahoo.com)

The Big Data Research Center, Henan University.
