#!/usr/bin/python
#-*- coding: UTF-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def adaboost_error(X,Y,tree,alpha):
    #adaboost：returns the final error rate of a whole adaboost
    N = Y.shape[0]
    C = np.unique(Y).shape[0]
    vote = np.zeros((C,N))

    for i in range(alpha.shape[0]):
        if(alpha[i] == 0):
            break
        L_p = tree[i].predict(X)
        for j in range(N):
            vote[L_p[j],j] = vote[L_p[j],j] + alpha[i]
    pre_label = np.argmax(vote,axis=0)
    e = float(sum(pre_label!=Y))/Y.shape[0]
    return e