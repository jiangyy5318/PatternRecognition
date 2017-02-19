#!/usr/bin/python
#-*- coding: UTF-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math
from adaboost_error import adaboost_error

def adaboost(train_X,train_Y,test_X,test_Y,maxIter):
    """
    :param train_X:N*784
    :param train_Y:N
    :param test_X:M*784
    :param test_Y:M
    :param maxIter:
    :return:
    """
    featureselect = np.random.permutation(train_X.shape[1])
    train_X = train_X[:,featureselect[0:20]]
    test_X = test_X[:,featureselect[0:20]]
    W = np.ones(train_Y.shape,dtype=float)/train_Y.shape[0] #initialize
    tree = []
    alpha = np.zeros(maxIter)
    e_train = np.zeros(maxIter)
    e_test = np.zeros(maxIter)

    for i in range(maxIter):
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(train_X,train_Y,sample_weight=W)
        tree.append(clf)
        L_p = clf.predict(train_X)
        e = float((L_p!=train_Y).sum())/train_Y.shape[0]
        print e
        alpha[i] = math.log((1-e)/e)
        print (L_p!=train_Y).sum()
        W = W * np.exp(alpha[i]*(L_p!=train_Y))
        print W
        W = W / W.sum()

        e_train[i] = adaboost_error(train_X,train_Y,tree,alpha)
        e_test[i] = adaboost_error(test_X,test_Y,tree,alpha)

    return e_train,e_test,maxIter