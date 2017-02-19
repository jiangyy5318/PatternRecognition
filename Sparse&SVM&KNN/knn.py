#!/usr/bin/python
#-*- coding: UTF-8 -*-
from scipy.spatial.distance import cdist
from numpy import *
import numpy as np
import scipy.io as sio

def knn(train_X,train_Y,test_X,test_Y,k,metric='euclidean'):
    #suggest k set 1,3,5,7
    #parameter refer to metric of cdist.:‘euclidean’, ‘hamming’,‘cityblock’
    dist = cdist(test_X ,train_X, 'euclidean')
    index = np.argsort(dist,axis=1,kind='quicksort')[:,0:k]
    pred = np.zeros(index.shape[0])
    for i in range(index.shape[0]):
        pred[i] = np.bincount(train_Y[index[i,:]]).argmax()
    succ = float((pred==test_Y).sum())/test_Y.shape[0]
    #print 'arcuracy = %f,k = %d, metric = %s'%(succ,k,metric)
    return succ
    data=sio.loadmat('mnist.mat')

if __name__ == "__main__":
     # read data
    train_X = data['train_X']
    train_Y = data['train_Y']
    test_X = data['test_X']
    test_Y = data['test_Y']
    if False:
        train_Num = 2000
        test_Num = 2000
        train_Select = np.random.permutation(train_X.shape[0])[0:train_Num]
        test_Select = np.random.permutation(test_X.shape[0])[0:test_Num]
        print knn(train_X[train_Select,:], train_Y[0,train_Select], test_X[test_Select,:], test_Y[0,test_Select], 3, metric='euclidean')
    if False:
        """
        arcuracy = 0.432500,train_Num = 20
        arcuracy = 0.752000,train_Num = 200
        arcuracy = 0.916000,train_Num = 2000
        arcuracy = 0.964500,train_Num = 20000
       """
        train_Num = [20,200,2000,20000]
        test_Num = 2000
        test_Select = np.random.permutation(test_X.shape[0])[0:test_Num]
        for e in train_Num:
            train_Select = np.random.permutation(train_X.shape[0])[0:e]
            succ = knn(train_X[train_Select,:], train_Y[0,train_Select], test_X[test_Select,:], test_Y[0,test_Select], 3, metric='euclidean')
            print 'arcuracy = %f,train_Num = %d'%(succ,e)
    if True:
        """
        arcuracy = 0.894000,K = 1
        arcuracy = 0.896000,K = 3
        arcuracy = 0.897500,K = 5
        arcuracy = 0.894500,K = 7
        arcuracy = 0.892500,K = 9
       """
        K = [1,3,5,7,9]
        train_Num = 2000
        test_Num = 2000
        train_Select = np.random.permutation(train_X.shape[0])[0:train_Num]
        test_Select = np.random.permutation(test_X.shape[0])[0:test_Num]
        for e in K:
            succ = knn(train_X[train_Select, :], train_Y[0, train_Select], test_X[test_Select, :], test_Y[0, test_Select], e,metric='euclidean')
            print 'arcuracy = %f,K = %d' % (succ, e)