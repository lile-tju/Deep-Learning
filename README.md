# Deep-Learning

The following contains
 - A MATLAB implementation of Convolutional Classification RBM
 - A Theano implementation of Classification RBM which can be used as 
   a top layer in DBN to obtain better performance than DBN
 - Instructions for running code in README.txt in each folder  
 - Thesis containing the results of above implementation

Abstract:

Deep Learning is an area of Machine Learning which consist of algorithms to model high level representation of data by use of multiple hidden layers. Restricted Boltzmann Machine (RBM) were developed to model input data distribution and were used as feature extractors for various classification algorithms. Deep Belief Networks (DBN) are stacked representation of RBM's which are pre-trained greedily to initialize multi layered neural network which is then fine tuned using back propagation.

Restricted Boltzmann Machine can be modified by augmenting an output layer above hidden layer known as Classification RBM (ClassRBM). It is then used to model input distribution along with its class labels. This architecture based on supervised learning can also be used as a standalone classifier. In this work, we pretrained Deep Belief Network greedily in unsupervised manner with ClassRBM as the top layer and fine-tuned to obtain better accuracy over traditional DBN. We argue that the better performance comes from pre training of weights between stacks of RBM and output layer which were randomly initialized previously for fine-tuning. Also, we introduce Convolutional Classification Restricted Boltzmann Machine, which is an extension of ClassRBM to  incorporate convolutional aspect and obtained better classification accuracy over ClassRBM.

