## Deep-4mCW2V

### Abstract
N4-methylcytosine is a kind of DNA modification which could regulate multiple biological processes such as transcription regulation, DNA replication and gene expressions. Correctly identifying 4mC sites in genomic sequences can provide precise knowledge about their genetic roles. This study aimed to develop a deep learning-based model to predict 4mC sites in the E. coli. In the proposed model, DNA sequences were encoded by word embedding technique ‘word2vec’. The obtained features were inputted into 1D convolutional neural network (CNN) to classify 4mC from non-4mC sites in Escherichia coli. On the independent dataset, our model could yield the overall accuracy of 0.861%, which was approximately 4.3% higher than the existing model, 4mCCNN respectively.

![alt tag](https://github.com/linDing-groups/Deep-4mCW2V/blob/W2V/Deep%20learning.png)

## Required Packages

    Python3 (tested 3.5.4)
    jupyter (tested 1.0.0)
    scikit-learn (tested 0.22.1)
    pandas (tested 1.0.1)
    numpy (tested 1.18.1)
    gensim (tested 3.8.1)
    sklearn (tested 0.19.1)
    keras (tested 2.3.1)
    tensorflow (tested 2.1.0)
    
## For Feature Generation
    W2V.py

## For Train the Model
    Train_CNN_Model.py

## Loading the Model
    Test.py
## Note
For files with different input sequences, you need to pay attention to the modification of parameters in code.
## Citation:
