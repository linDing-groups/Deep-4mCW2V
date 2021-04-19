#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from keras.models import load_model
import keras.backend as K
from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Reshape, normalization
from keras.models import Model
from keras.utils import to_categorical
from keras.layers.recurrent import LSTM
from sklearn import metrics
import random
from keras.models import model_from_json

#define evaluation indicators
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(test_Y, pre_test_y):
     #calculate the F1-score
    Precision = precision(test_Y, pre_test_y)
    Recall = recall(test_Y, pre_test_y)
    f1 = 2 * ((Precision * Recall) / (Precision + Recall + K.epsilon()))
    return f1 

def TP(test_Y,pre_test_y):
    #calculate numbers of true positive samples
    TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
    return TP

def FN(test_Y,pre_test_y):
     #calculate numbers of false negative samples
    TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
    P=K.sum(K.round(K.clip(test_Y, 0, 1)))
    FN = P-TP #FN=P-TP
    return FN

def TN(test_Y,pre_test_y):
    #calculate numbers of True negative samples
    TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
    return TN

def FP(test_Y,pre_test_y):
    #calculate numbers of False positive samples
    N = (-1)*K.sum(K.round(K.clip(test_Y-K.ones_like(test_Y), -1, 0)))#N
    TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
    FP=N-TN
    return FP
  
data = np.array(pd.read_csv('/content/drive/MyDrive/222_vecs.csv')) #testing set
    
X1 = data[0:215, 1:] #215 is the number of positive samples in testing set, '1' is the label of positive sample
Y1 = data[0:215, 0] #'0' is the label of negative sample
X2 = data[215:, 1:]
Y2 = data[215:, 0]
X = np.concatenate([X1, X2], 0)
Y = np.concatenate([Y1, Y2], 0)

#Y = Y.reshape((Y.shape[0], -1))
X = np.expand_dims(X, 2)
print (X)
print ("X.shape: ", X.shape)
print ("Y.shape: ", Y.shape)
model_name = ('/content/drive/MyDrive/Deep-4mCW2V/Deep-4mCW2V_model/saved_model.h5') #load model generated by train_CNN_model.ipynb
model_back = load_model(model_name, 
                        custom_objects={'precision': precision,'recall':recall,'f1':f1,'TP':TP,'FN':FN,'TN':TN,'FP':FP})
# model = load_model('/content/drive/MyDrive/Deep-4mCW2V/Deep-4mCW2V_model/saved_model.h5')
accuracy = model_back.evaluate(X,Y)
# print 'loss', loss
print ('accuracy', accuracy) 
maxprobability = model_back.predict(X)
np.set_printoptions(threshold=np.inf)
print ('maxprobability') #print maxprobability
fw = open("//content/drive/MyDrive/Deep-4mCW2V/Result.txt", "w") #define result outputFile 
myprob = "\n".join(map(str, maxprobability[:, 0]))
fw.write(myprob)
predictclass = model_back.predict(X)
predictclass = np.argmax(predictclass,axis=1)
print ('predictclass')
