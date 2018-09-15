Multi_Imbalance (Implementations in Octave)

Implementations of state-of-the-art Multi-class Imbalance learning algorithms.

Copyright: Jingjun Bi and Chongsheng Zhang. 
The two contributors of this open software for multi-class imbalance learning are  Ms. Jingjun Bi and Prof. Chongsheng Zhang.


*(1) Software Contents
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



*(2) User Manual and Usage Examples

There are 7 classes (categories) of algorithms for multi-class imbalance learning, each class consisting of one or more algorithms.  In total, there are 18 major algorithms for multi-class imbalance learning.

See User_manual_Octave.pdf for user manuals and examples of the main functionality of the software.
