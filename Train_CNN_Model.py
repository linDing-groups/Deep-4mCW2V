#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Reshape, normalization
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from keras.layers.recurrent import LSTM
import h5py
from sklearn import metrics
import random
from keras.regularizers import l1, l2
from tensorflow.python.keras.optimizers import TFOptimizer

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

def dnn_model(train_X, train_Y, test_X, test_Y, lr, epoch, batch_size):
    train_X = np.expand_dims(train_X, 2)
    test_X = np.expand_dims(test_X, 2)
    inputs = Input(shape = (train_X.shape[1], train_X.shape[2]))
    x = Conv1D(32, kernel_size = 2, strides = 1, padding = 'valid', activation = 'relu')(inputs)
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation = 'relu',kernel_regularizer = l2(1e-7))(x)
    x = Dense(16, activation = 'relu',kernel_regularizer = l2(1e-7))(x)
    x = Dense(8, activation = 'relu',kernel_regularizer = l2(1e-7))(x)
    predictions = Dense(1, activation = 'sigmoid')(x) 
    model = Model(inputs = inputs, outputs = predictions)
    print("model")
    model.compile(optimizer = 'RMSProp',
                  loss = 'mean_squared_error',
                  metrics = ['acc',precision,recall,f1,TP,FN,TN,FP])
    print("compile")
    result = model.fit(train_X, train_Y, epochs = epoch, batch_size = 32, validation_data = (test_X, test_Y), shuffle = True)
    model.save('Deep-4mCW2V.h5') #save model
    pre_test_y = model.predict(test_X, batch_size = 50)
    pre_train_y = model.predict(train_X, batch_size = 50)
    test_auc = metrics.roc_auc_score(test_Y, pre_test_y)
    train_auc = metrics.roc_auc_score(train_Y, pre_train_y)
    print("train_auc: ", train_auc)
    print("test_auc: ", test_auc) 
    return test_auc, result

# split data and output result
data = np.array(pd.read_csv('222_vecs.csv'))#inputfile
X1 = data[0:215, 1:]#215 is the number of positive samples in training set, '1' is the label of positive sample
Y1 = data[0:215, 0]#'0' is the label of negative sample
X2 = data[215:, 1:]
Y2 = data[215:, 0]
X = np.concatenate([X1, X2], 0)
Y = np.concatenate([Y1, Y2], 0)
#Y = Y.reshape((Y.shape[0], -1))
print (X)
print ("X.shape: ", X.shape)
print ("Y.shape: ", Y.shape)

lr = 0.05 #learning rate
epoch = 50
batch_size = 32
kf = KFold(n_splits = 5, shuffle = True, random_state = 20)
#kf = KFold(n_splits = 5, shuffle = False)
kf = kf.split(X)

test_aucs = []
for i, (train_fold, validate_fold) in enumerate(kf):
    print("\n\ni: ", i)
    test_auc = dnn_model(X[train_fold], Y[train_fold], X[validate_fold], Y[validate_fold], lr, epoch, batch_size)
    test_auc = test_aucs.append(test_auc)
w = open("//content/drive/MyDrive/Deep-4mCW2V/Train.txt", "w")
for j in test_aucs: 
    w.write(str(j) + ',')
w.write('\n')
w.write(str(np.mean(test_aucs)) + '\n')
w.close()
