# Multi_Imbalance
Our implementations of the Multi-class Imbalance learning algorithms (for the KBS paper)

# Copyright: The main contributor of these implementations is Ms. Jingjun Bi.  

*Note:The users should run these codes in the Windows Operating Systems. If you run the codes with Mac OSx, there will be problems with the LibSVM software package we have included. Thus we suggest the readers to use the Windows OS. 

*If you have any problems, please do not hesitate to send us an email: 2834335964@qq.com; henucs@qq.com

Before running the program, please add the folder 'test' and all subfolders to the path.

(1). The folder 'codeBoost' contains 5 Boosting methods：AdaBoost.M1, AdaC2.M1, SAMME, AdaBoost.NC, PIBoost

(2). The folder 'codeDOVO' contains algorithm DOVO

(3). The folder 'codeHDDT' contains 3 methods: MC-HDDT, HDDT+ECOC, HDDT+OVA

(4). The folder 'codeimECOC' contains 3 methods: imECOC+sparse, imECOC+dense, imECOC+OVA

(5). The folder 'codeMultiIM' contains 4 methods: Multi-IM+OVO, Multi-IM+A&O, Multi-IM+OAHO, Multi-IM+OVA

(6). The folder 'codeDECOC' contains algorithm DECOC

(7). The folder 'codeFuzzyImb' contains algorithm IFROWANN+ECOC

A total of 18 algorithms. 


The folder 'data' is the experimental datasets, and each experiment is carried out using 5-fold cross validation

The folder 'results' are the experimental results, '_c.mat' is time consuming, and '_p.mat' is prediction labels . 


The file 'testall.m' is an example of all the methods tested, a total of 19 methods including the above 18 algorithms and the base classifier CART.


Input: data(d).train, data(d).trainlabel, data(d).test 
         ---training data matrix, training labels, test data matrix
Output:Cost(d).NAME, Pre(d).NAME
         ---time consuming, prediction labels
