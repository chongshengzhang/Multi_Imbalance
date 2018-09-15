# Multi_Imbalance
Implementations of 18 major (state-of-the-art) Multi-class Imbalance learning (Imbalance Classification) algorithms.

# Copyright: Jingjun Bi and Chongsheng Zhang (The Big Data Research Center, Henan University).

*The two contributors of this open software for multi-class imbalance learning are  Ms. Jingjun Bi and Prof. Chongsheng Zhang (chongsheng.zhang@yahoo.com, or, henucs@qq.com). 
* Ms. Bi mainly contributes to the  implementations of these 18 algorithms; 
* Prof.Zhang is the supervisor of Ms. Bi, he proposed the idea for this work, 
* he also throughly investigated the state-of-the-art and guided Ms. Bi to implement these algorithms.
* Moreover, Prof.Zhang made the majority annotations and the user manual for this software; 
* he also writes the paper (with other co-authors) for the software.
* Note: Anyone who use this software should refer to (should cite) the following papers:
* Note: This software is protected by the GNU General Public License (GPL). 
        For academic usage, the users should always cite the following two papers; 
        For commerical usage, the company or institution (and any project) must have the consent from 
        Prof. Chongsheng Zhang (henucs@qq.com) first.

Please cite the following two papers:

-Jingjun Bi, Chongsheng Zhang*. (2018). An Empirical Comparison on State-of-the-art Multi-class Imbalance
 Learning Algorithms and A New Diversified Ensemble Learning Scheme.
 Knowledge-based Systems, 2018, Vol. 158, pp. 81-93.

-To add Our coming KBS Open Software Paper.


*If you have any problems, please do not hesitate to send us an email: henucs@qq.com

# (1) Installation

Software Environments: Please first install and use the Matlab or OCTAVE softwares. 

Please download the software to your local disk, then add the folder 'Multi_Imbalance' and all subfolders to the path of Matlab or OCTAVE.

* The users can run these codes in the Windows Operating Systems or Mac OSx. 

* If you run the codes with Mac OSx, there will be issues with the LibSVM software package we have included. In this case, users just ignore (stop using) LibSVM.


# (2) Software Contents

The folder 'Boost' contains a class of 5 Boosting methods: AdaBoost.M1, AdaC2.M1, SAMME, AdaBoost.NC, PIBoost).

The folder 'DECOC' contains algorithm DECOC.

The folder 'DOVO' contains the algorithm DOVO.

The folder 'FuzzyImb' contains algorithm FuzzyImb+ECOC.

The folder 'HDDT' contains a class of 3 methods: MC-HDDT, HDDT+ECOC, HDDT+OVA.

The folder 'imECOC' contains a class of 3 methods: imECOC+sparse, imECOC+dense, imECOC+OVA.

The folder 'MultiIM' contains a class of 4 methods: Multi-IM+OVO, Multi-IM+A&O, Multi-IM+OAHO, Multi-IM+OVA.


In total,  18 algorithms for multi-class imbalance learning.

(Note that we also include the 19th algorithm CART as the baseline method, but CART is not towards imbalance learning.)


The folder 'data' is the experimental datasets, and each experiment is carried out using 5-fold cross validation. 

The folder 'results' are the experimental results, '_c.mat' is time consuming, and '_p.mat' is prediction labels.

The file 'testall.m' is an example of all the methods tested, a total of 19 methods including the above 18 algorithms and the base classifier CART. 

Input: data(d).train, data(d).trainlabel, data(d).test, training labels, test data matrix  (the training data and testing data matrix)

Output:Cost(d).NAME, Pre(d).NAME, prediction labels (---time consumptions)

In the following, we will give details of these 18 major algorithms for multi-class imbalance learning.



# (3) Software Usage Manual

There are 7 classes (categories) of algorithms for multi-class imbalance learning, each class consisting of one or more algorithms. 

In total, there are 18 major algorithms for multi-class imbalance learning.

In the following, we give the user manual of these 18 major algorithms for multi-class imbalance learning.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# 1. Variants of AdaBoost for Multi-class Imbalance Learning 
     (a class of 5 algorithms in total, all of which are variants of Boost/AdaBoost)

% These 5 algorithms are under the folder "Boost".

%

% AdaBoost (Adaptive Boosting) is a binary classification algorithm proposed by Freund and Schapire that

% integrates multiple weak classifiers to build a stronger classifier. AdaBoost only supports binary data in

% the beginning, but it was later extended to multi-class scenarios. AdaBoost.M1 and SAMME (Stagewise

% Additive Modeling using a Multi-class Exponential loss function) have extended AdaBoost in both the update

% of samples’ weights and the classifier combination strategy. The main difference between them is the method

% for updating the weights of the samples.

% (1) AdaBoost.M1. The main steps of AdaBoost.M1 are as follows:

% Step 1: Initialize the weight Vector with uniform distribution

% Step 2: for t=1 to Max_Iter do

% Step 3:    Fit a classifier nb to the training data using weights

% Step 4:    Compute weighted error

% Step 5:    Compute AlphaT=0.5*log((CorrectRate+eps)/(errorRate+eps))

% Step 6:    Update weights

% Step 7:    Re-normalize weight

% Step 8: end for

% Step 9: Output Final Classifier

% Reference for AdaBoost.M1:

% Freund, Y. & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an
  application to boosting. Journal of Computer and System Sciences, August 1997, 55(1).

%

% (2) SAMME. The main procedure (steps) of the SAMME algorithm:

% Step 1: Initialize the weight Vector with uniform distribution

% Step 2: for t=1 to Max_Iter do

% Step 3:    Fit a classifier nb to the training data using weights

% Step 4:    Compute weighted error: errorRate=sum(weight(find(predicted~=trainlabel)));

% Step 5:    Compute AlphaT=log((1-errorRate)/(errorRate+eps))+log(length(labels)-1)

% Step 6:    Update weights weight(i)=weight(i)* exp( AlphaT(t));(trainlabel(i)~=predicted(i))

% Step 7:    Re-normalize weight

% Step 8: end for

% Step 9: Output Final Classifier

% Reference for SAMME:

% Zhu, J., Zou, H., Rosset, S., et al. (2006). Multi-class AdaBoost. Statistics & Its Interface,
  2006, 2(3), 349-360.

% (3) AdaC2.M1 (adaC2cartM1). It derives the best cost setting through the genetic algorithm (GA) method, then

% takes this cost setting into consideration in the subsequent boosting. Genetic algorithm proposed by

% Holland is based on natural selection and genetics of random search technology. GA can achieve

% excellent performance in finding the best parameters.

% Reference for AdaC2.M1:

% Sun, Y., Kamel, M. S. & Wang, Y. (2006). Boosting for learning multiple classes with imbalanced class
  distribution. Proceedings of the 6th International Conference on Data Mining, 2006 (PP. 592-602).
  
% (4) AdaBoost.NC. Since GA is very time consuming, in the above reference, the authors propose AdaBoost.NC which

% deprecates the GA algorithm, but emphasizes ensemble diversity during training, and exploits its

% good generalization performance to facilitate class imbalance learning.

% Reference for AdaBoost.NC:

% Wang, S., Chen, H. & Yao, X. Negative correlation learning for classification ensembles. Proc. Int. Joint
  Conf. Neural Netw., 2010 (PP. 2893-2900).

% (5) PIBoost. It combines binary weak-learners to separate groups of classes, and uses a margin-based

% exponential loss function to classify multi-class imbalanced data.

% Reference for PIBoost:

% Fernndez, B. A. & Baumela. L. (2014). Multi-class boosting with asymmetric binary weak-learners. 
  Pattern Recognition, 2014, 47(5), PP. 2080-2090.
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.1 AdaBoost.M1
    % The main function is: adaboostcartM1()
    % The principles and procedure (and the rational) of this algorithm are explained/given in adaboostcartM1.m.
			
    % Usage for adaboostcartM1:
		
    for d=1:5
   
        [Cost(d).adaboostcartM1tr,Cost(d).adaboostcartM1te,Pre(d).adaboostcartM1] = adaboostcartM1(data(d).train,data(d).trainlabel,data(d).test,20);
     
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.2 SAMME
    % The main function is: SAMMEcart()
    % The principles and procedure (and the rational) of this algorithm are explained/given in SAMMEcart.m.
			
    % Usage for adaboostcartM1:
		
    for d=1:5
       
        [Cost(d).SAMMEcarttr,Cost(d).SAMMEcartte,Pre(d).SAMMEcart] = SAMMEcart(data(d).train,data(d).trainlabel,data(d).test,20);
      
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.3 AdaC2.M1
    % The main function is: adaC2cartM1()
    % The principles and procedure (and the rational) of this algorithm are explained/given in adaC2cartM1.m.
			
    % Usage for adaC2cartM1:
		
    for d=1:5
        tic;
        C0=GAtest(data(d).train,data(d).trainlabel);
        Cost(d).GA=toc;
        Indx(d).GA=C0;
       
        [Cost(d).adaC2cartM1GAtr,Cost(d).adaC2cartM1GAte,Pre(d).adaC2cartM1GA] = adaC2cartM1(data(d).train,data(d).trainlabel,data(d).test,20,C0);
     
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.4 AdaBoost.NC
    % The main function is: adaboostcartNC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in adaboostcartNC.m.
			
    % Usage for adaboostcartNC:
		
    for d=1:5
 
        [Cost(d).adaboostcartNCtr,Cost(d).adaboostcartNCte,Pre(d).adaboostcartNC] = adaboostcartNC(data(d).train,data(d).trainlabel,data(d).test,20,2);
    
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.5 PIBoost
    % The main function is: PIBoostcart()
    % The principles and procedure (and the rational) of this algorithm are explained/given in PIBoostcart.m.
			
    % Usage for PIBoostcart:
		
    for d=1:5
     
        [Cost(d).PIBoostcarttr,Cost(d).PIBoostcartte,Pre(d).PIBoostcart] = PIBoostcart(data(d).train,data(d).trainlabel,data(d).test,20);
      
    end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# 2. DECOC 
     (This algorithm is our proposed ensemble approach for imbalance learning, DOVO+imECOC)

% This algorithm is under the folder "DECOC".

% Reference (we propose and implement the following algorithm):

%   Jingjun Bi, Chongsheng Zhang*. (2018). An Empirical Comparison on State-of-the-art Multi-class Imbalance
    Learning Algorithms and A New Diversified Ensemble Learning Scheme.
    Knowledge-based Systems, 2018, Vol. XXX, pp. XXX.

%(1) Using funclassifierDECOC, DECOC uses ECOC strategy to tranform the multi-class data into multiple binary data,

%    then finds the best classifier for each specific binaried data, which will be kept by ft.

%(2) Using funcwEDOVO, it builds the best classifier for each binarized data (by ECOC) and the predictions are in allpre.

%    Notice that, in this function, it has duplicates with funclassifierDECOC, in building the best classification

%    model for each specific binarized data.  In specific, funcwEDOVO calls funcPreEDOVO,

%    which (retrains) rebuilds the model that funclassifierDECOC has built previously. 

%    This is for the traning data.


%(3) With funcpretestEDOVO  to make the predictions on the test data, using the imECOC algorithm. 


# 2.1 DECOC
    % The main function is: DECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in DECOC.m.
	
    % Usuage of the DECOC method: 
		
    for d=1:5
  
        [Cost(d).imECOCDOVOs1tr,Cost(d).imECOCDOVOs1te,Pre(d).imECOCDOVOs1] = DECOC(data(d).train,data(d).trainlabel,data(d).test, 'sparse',1);
      
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






# 3. DOVO 
     (This algorithm is an ensemble based approach for imbalance learning)

% This algorithm is under the folder "DOVO".


% Reference (we implement the following algorithm):
  Kang, S., Cho, S. & Kang P. (2015) Constructing a multi-class classifier using one-against-one approach
   with different binary classifiers. Neurocomputing, 2015, Vol. 149, pp. 677-682. 

% The main idea of this algorithm:

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

# 3.1 DOVO
    % The main function is: DOVO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in DOVO.m.
	
    % Usuage of the DOVO method: 
    for d=1:5

        [Cost(d).DOAOtr,Cost(d).DOAOte,Pre(d).DOAO,Indx(d).C] = DOAO([data(d).train,data(d).trainlabel],data(d).test,data(d).testlabel,5);

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






# 4. fuzzyImbECOC 
     (This algorithm is a combination of FuzzyImb and ECOC, i.e., FuzzyImb+ECOC)

% This algorithm is under the folder "fuzzyImbECOC".



% Reference: E. Ramentol, S. Vluymans, N. Verbiest, et al. , IFROWANN: Imbalanced Fuzzy-Rough Ordered Weighted
             Average Nearest Neighbor Classification, IEEE Transactions on Fuzzy Systems 23 (5) (2015) 1622-1637.

% Note 1: We obtain the codes of IFROWANN (fuzzyImb) from the authors, we greatly acknowledge their help and contributions;

% Note 2: IFROWANN was originally designed for binary imbalanced data.

%         In this work, we extend IFROWANN with the ECOC encoding strategy to handle multi-class imbalanced data.


%(1) using funECOC(), it first generates the ECOC matrix (with each codeword for a specific class).

%    each class will be represented by an array of codes such as 1 1 -1 -1 1 -1.

%(2) it then extracts the instances (and the corresponding labels) of each original class, keep in train{i};

%(3) the ECOC matrix for all the classes is an nc*number1 matrix, each row represents the codeword of one class.

%    a)  for each column of the ECOC matrix, first retrieve the corresponding bit value of each class.

%        then assign this bit value as the label for all the instances of the current class.

%        This is for handling the multi-class data.

%
%    b)  for each two-class data, use fuzzyImb to train the binary classifier (see above reference).

%    at the end, we will train a few binary classifiers.


%    c)  for each test instance from testdata, use all the classifiers obtained from b) to make predictions;

%    their predicitions will be  combined as an array, 

%    then use the ECOC decoding method to find the nearest ECOC codeword, then the  corresponding class label.

# 4.1 fuzzyImbECOC
    % The main function is: fuzzyImbECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in fuzzyImbECOC.m.
			
    % Usage for fuzzyImbECOC (FuzzyImb+ECOC)
    for d=1:5
      tic;
			
      [Pre(d).fuzzyw6] = fuzzyImbECOC(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel, 'w6',0.1); 
			
      Cost(d).fuzzyw6=toc;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





# 5. HDDT 
     (a class of 3 algorithms in total, all of which are variants of HDDT)

% These 3 algorithms are under the folder "HDDT".

%

% Reference：
  Hoens, T. R., Qian, Q., Chawla, N. V., et al. (2012). Building decision trees for the multi-class imbalance problem.   
  Advances in Knowledge Discovery and Data Mining. Springer Berlin Heidelberg, 2012 (PP. 122-134).
  
% HDDTova, Decomposition Techniques OVA + HDDT, for multi-class imbalanced data.

% This is our own extension of HDDT to multi-class imbalanced data.

% it builds numberc of binary HDDT classifiers by combining the OVA strategy and HDDT.

% then combines the outputs of different binary HDDT classifiers generated using the OVA strategy,

% here, the decoding strategy for OVA is the same as the imECOC decoding. 

% It must be noted that the decoding strategy for testHDDTecoc and testHDDTova are identical,

% for fair comparisons between them.

% HDDTecoc, Decomposition Techniques ECOC + HDDT, for multi-class imbalanced data.

% This is our own extension of HDDT to multi-class imbalanced data, using the ECOC approach.

% it builds number1 of binary HDDT classifiers by combining the ECOC strategy and HDDT.

% then combines the outputs of different binary HDDT classifiers generated using the ECOC strategy,

% here, the decoding strategy for ECOC is the same as the imECOC decoding.

% It must be noted that the decoding strategy for testHDDTecoc and testHDDTova are identical,

% for fair comparisons between them.

% MCHDDT, the Multi-Class HDDT method, successively takes one or a pair of classes as the positive class and the

% rest as negative class, when calculating the Hellinger distance for each feature. It next selects the

% maximum Hellinger value for this feature. Finally, it obtains the maximum Hellinger value for each feature,

% then the feature with maximum Hellinger distance will be used to split the node.

% Then, after determining the best split feature, it recursively build the child trees.

% It should be noted that, the Principles of HDDT are described as follows:

% The Hellinger distance decision trees (HDDT) method is a classification algorithm based on decision trees

% for binary imbalanced data. When building a decision tree, the splitting criterion used in the HDDT is the

% Hellinger distance, see the formula (1) in our KBS paper, where |X_+| and |X_-|) respectively represents

% the total number of samples with positive (negative) labels, whereas |X_(+j)| and X_(-j)| respectively

% denotes the number of positive (negative) examples with the j-th value of the current feature.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 5.1 HDDTova
    % The main function is: HDDTova()
    % The principles and procedure (and the rational) of this algorithm are explained/given in HDDTova.m.
			
    % Usage for HDDTova:
		
    for d=1:5
    
        [Cost(d).HDDTovatr,Cost(d).HDDTovate,Pre(d).HDDTova] = HDDTova(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);
        
    end
		

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

# 5.2 HDDTecoc
    % The main function is: HDDTecoc()
    % The principles and procedure (and the rational) of this algorithm are explained/given in HDDTecoc.m.
			
    % Usage for HDDTecoc:
		
    for d=1:5

        [Cost(d).HDDTecoctr,Cost(d).HDDTecocte,Pre(d).HDDTecoc] = HDDTecoc(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);

    end
		

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# 5.3 MCHDDT
    % The main function is: MCHDDT()
    % The principles and procedure (and the rational) of this algorithm are explained/given in MCHDDT.m.
			
    % Usage for MCHDDT:
		
    for d=1:5
  
        [Cost(d).MCHDDTtr,Cost(d).MCHDDTte,Pre(d).MCHDDT] = MCHDDT(data(d).train,data(d).trainlabel,data(d).test,data(d).testlabel);
    
    end		

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






# 6. imECOC 
     (a class of 3 algorithms in total, all of which are variants of imECOC)

% These 3 algorithms are under the folder "imECOC".

% Reference:

% Liu, X. Y., Li, Q. Q. & Zhou Z H. (2013). Learning imbalanced multi-class data with optimal dichotomy
  weights. IEEE 13th International Conference on Data Mining (IEEE ICDM), 2013 (PP.  478-487). 

see http://cs.nju.edu.cn/_upload/tpl/01/0b/267/template267/zhouzh.files/publication/icdm13imECOC.pdf


% The imECOC algorithm includes the following techniques:

% (1) in each binary classifier, it simultaneously considers the between-class and the within-class

%     imbalance; see function funclassifier();

% (2) in the training/prediction phase, it assigns different weights to different binary classifiers;

%     see function funcw();

% (3) in the prediction phase, it decodes it with weighted distance to obtain the optimal weight of the

%     classifier by minimizing the weighted loss.  see function funcpre().

% In our implementations, Sparse, Dense, OVA are the 3 different coding methods for ECOC, 

% OVA is actually a special case of ECOC. 

% Actually, ECOC has many other coding strategies.


% It should be noted that,

% The ECOC (Error Correcting Output Codes) decomposition method, uses the idea of error correction output coding to classify  

% the multi-class data. ECOC can adopt different encoding and decoding methods (such as Dense, Sparse, OVA, etc.). 

% ECOC first builds a codeword for each class to obtain the largest distance (such as Hamming 

% distance) between various classes, thus transforms the classes of the multi-class data into c codewords 

(when using OVA as the encoding method).
   
% In the testing phase, 

% ECOC uses these c classifiers to respectively predict the test sample, then obtains a combined output. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 6.1 imECOC+sparse
    % The main function is: imECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in imECOC.m.
			
    % Usage for imECOC+sparse:
		
    for d=1:5
    
        [Cost(d).imECOCs1tr,Cost(d).imECOCs1te,Pre(d).imECOCs1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'sparse',1);
       
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

# 6.2 imECOC+OVA
    % The main function is: imECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in imECOC.m.
			
    % Usage for imECOC+OVA:
		
   for d=1:5
     
        [Cost(d).imECOCo1tr,Cost(d).imECOCo1te,Pre(d).imECOCo1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'OVA',1);
        
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
		
# 6.3 imECOC+dense
    % The main function is: imECOC()
    % The principles and procedure (and the rational) of this algorithm are explained/given in imECOC.m.
			
    % Usage for imECOC+dense:
		
    for d=1:5
      
        [Cost(d).imECOCd1tr,Cost(d).imECOCd1te,Pre(d).imECOCd1] = imECOC(data(d).train,data(d).trainlabel,data(d).test, 'dense',1);
    
    end
		
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






# 7. MultiIM 
    (a class of 4 algorithms in total, all of which are variants of Multi-IM)

% These 4 algorithms are under the folder "MultiIM".

% Reference:

  Ghanem, A. S., Venkatesh, S. & West, G. (2010). Multi-class pattern classification in imbalanced data.
  International Conference on Pattern Recognition (ICPR), 2010 (PP. 2881-2884).
  

% PRMs-IM is a classification algorithm (originally designed) for binary imbalanced data.

% Let m be the ratio between the number of majority samples and that of the minority samples. PRMs-IM

% randomly divides the majority samples into m parts, next combines each part with all the minority

% instances, then trains a corresponding binary classifier.  In the prediction phase, it uses weighted voting

% to ensemble the outputs of the m classifiers and makes the final prediction.


% The above Reference (the above paper) proposes the Multi-IM algorithm which 

% combines A&O and PRMs-IM, where PRMs-IM is adopted to train the classifier for A&O.

% Besides A&O, in our work, we also combine the OVA, OVO and OAHO decomposition methods

% with PRMs-IM to further investigate the performance of PRMs-IM for multi-class imbalance learning.


% It should be noted that:

% OVA: One-vs-All, also known as ‘One-against-all’, is a relatively simple decomposition strategy. For each class

(category), it labels this class as a ‘positive class’ 

% and all the other classes as ‘negative classes’, then trains a corresponding classification model. This way, multi-class

classification is transformed into multiple 

% binary classification problems. If the original data has c classes (categories), a total of c binary classifiers will be

learnt. 


% OVO: the One-vs-One decomposition strategy first selects a subset from the original data that only contains the instances

for each pair of classes, then trains a 

% binary classifier for each pair of classes, hence a total of c (c-1) / 2 binary classifiers will be obtained. In the

prediction phase, all the c (c-1) / 2 binary 

% classifiers will be used to predict a new instance, and their corresponding predictions will be combined using certain

rules to make the final prediction. 

% A&O: the All-and-One (A&O) method combines the advantages of OVA and OVO to avoid their shortcomings. 

% When predicting a new sample, A&O first uses OVA to get the top-2 prediction results (c_i,c_j), then 

% adopts the OVO classifier previously trained for the pair of classes containing c_i and c_j to make the final prediction.


% OAHO: One-Against-Higher-Order (OAHO) is a decomposition method specifically designed for imbalanced data. 

% OAHO first sorts the class by the number of samples in descending order. Let the sorted classes being {C_1,C_2,…,C_k},
with C_1 having the largest number of samples. 

% Starting from C_1 until C_(k-1), OAHO sequentially labels the current class as ‘positive class’ and all the rest classes

with lower ranks as ‘negative classes’, then 

% trains a binary classifier. Therefore, there will be k-1 binary classifiers in total. When predicting a new sample, the

first classifier is used to predict the sample, 

% if the prediction result is C_1, then outputs C_1 as the final result; otherwise, it switches to the second classifier to

make the prediction, and so on, until the final prediction result is obtained.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 7.1 Muti-IM+OVA
    % The main function is: classOVA()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classOVA.m.
			
    % Usage for Muti-IM+OVA:
		
    for d=1:5
     
        [Cost(d).classOVAtr,Cost(d).classOVAte,Pre(d).classOVA] = classOVA(data(d).train,data(d).trainlabel,data(d).test);
     
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

# 7.2 Muti-IM+OVO
    % The main function is: classOAO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classOVO.m.
		
    % Usage for Muti-IM+OVO:		
    
    for d=1:5
      
        [Cost(d).classOAOtr,Cost(d).classOAOte,Pre(d).classOAO] = classOAO([data(d).train,data(d).trainlabel],data(d).test);
    
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 7.3 Muti-IM+OAHO
    % The main function is: classOAHO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classOAHO.m.
		
    % Usage for Muti-IM+OAHO:		
        
    for d=1:5

        [Cost(d).classOAHOtr,Cost(d).classOAHOte,Pre(d).classOAHO] = classOAHO([data(d).train,data(d).trainlabel],data(d).test);
      
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 7.4 Muti-IM+A&O
    % The main function is: classAandO()
    % The principles and procedure (and the rational) of this algorithm are explained/given in classAandO.m
		
    % Usage for Muti-IM+A&O:	    
 
    for d=1:5
       
        [Cost(d).classAandOtr,Cost(d).classAandOte,Pre(d).classAandO] = classAandO(data(d).train,data(d).trainlabel,data(d).test);
      
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This is the END. Thank you so much.


Prof.Chongsheng Zhang (chongsheng.zhang@yahoo.com, or, henucs@qq.com), 

The Big Data Research Center, Henan University.
