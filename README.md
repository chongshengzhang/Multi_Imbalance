# Multi_Imbalance
Our implementations of the Multi-class Imbalance learning algorithms (for the KBS paper)

The main contributor of these implementations is Ms. Jingjun Bi.  

Before running the program, add the folder 'test' and all subfolders to the path.

The folder 'codeBoost' contains 5 Boosting methods：AdaBoost.M1, AdaC2.M1, SAMME, AdaBoost.NC, PIBoost

The folder 'codeDOVO' contains algorithm DOVO

The folder 'codeHDDT' contains 3 methods: MC-HDDT, HDDT+ECOC, HDDT+OVA

The folder 'codeimECOC' contains 3 methods: imECOC+sparse, imECOC+dense, imECOC+OVA

The folder 'codeMultiIM' contains 4 methods: Multi-IM+OVO, Multi-IM+A&O, Multi-IM+OAHO, Multi-IM+OVA

The folder 'codeDECOC' contains algorithm DECOC

The folder 'codeFuzzyImb' contains algorithm IFROWANN+ECOC

A total of 18 algorithms


The folder 'data' is the experimental datasets, and each experiment is carried out using 5-fold cross validation

The folder 'results' are the experimental results, '_c.mat' is time consuming, and '_p.mat' is prediction labels . 


The file 'testall.m' is an example of all the methods tested, a total of 19 methods including the above 18 algorithms and the base classifier CART.


Input: data(d).train, data(d).trainlabel, data(d).test 
         ---training data matrix, training labels, test data matrix
Output:Cost(d).NAME, Pre(d).NAME
         ---time consuming, prediction labels
