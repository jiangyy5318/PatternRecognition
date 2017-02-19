#!/usr/bin/python
#-*- coding: UTF-8 -*-
from scipy.spatial.distance import cdist
from numpy import *
import numpy as np


def knn(train_X,train_Y,test_X,test_Y,k,metric='euclidean'):
    #suggest k set 1,3,5,7
    #parameter refer to metric of cdist.:‘euclidean’, ‘hamming’,‘cityblock’
    dist = cdist(test_X ,train_X, 'euclidean')
    index = np.argsort(dist,axis=1,kind='quicksort')[:,0:k]
    pred = np.zeros(index.shape[0])
    for i in range(index.shape[0]):
        pred[i] = np.bincount(train_Y[index[i,:]]).argmax()
    succ = float((pred==test_Y).sum())/test_Y.shape[0]
    print 'arcuracy = %f,k = %d, metric = %s'%(succ,k,metric)
    return succ
