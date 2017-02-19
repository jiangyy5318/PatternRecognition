#!/usr/bin/python
#-*- coding: UTF-8 -*-
import numpy as np
from numpy.matlib import repmat
from numpy.linalg import inv
from scipy.sparse.linalg import eigsh
#from scipy.linalg import svd

def MultipleClasses(X,Y,d = 40):
    """

    """
    C = np.unique(Y)
    len = X.shape[1]
    S_B = np.zeros((len,len))
    S_W = np.zeros((len,len))
    m_total = np.mean(X,axis=0)
    for e in C:
        #temp X\in class i
        temp = X[np.where(Y==e)[0],:]
        m_class = np.mean(temp,axis=0)
        S_B = S_B + np.dot((m_class - m_total).transpose(),(m_class-m_total))
        S_W = S_W + np.dot((temp - repmat(m_class,temp.shape[0],1)).transpose(),(temp - repmat(m_class,temp.shape[0],1)))

    S_W_1S_W = np.dot(inv(S_W),S_B)
    t,v = eigsh(S_W_1S_W, k=d, which='LM')
    print v
    return v