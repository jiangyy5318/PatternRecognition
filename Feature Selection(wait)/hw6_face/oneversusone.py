#!/usr/bin/python
#-*- coding: UTF-8 -*-
import numpy as np
from numpy.matlib import repmat
from numpy.linalg import inv
from scipy.sparse.linalg import eigsh
#from scipy.linalg import svd

def oneversusone(X,Y,d = 40):
    """
    """
    C = np.unique(Y)
    len = X.shape[1]
    classes = C.shape[0]
    C_mean = np.zeros((classes,len))
    S_W = np.zeros((classes,len,len))
    weight = np.zeros((len,classes*(classes-1)/2))
    m_total = np.mean(X,axis=0)
    for i,e in zip(range(classes),C):
        #temp X\in class i
        temp = X[np.where(Y==e)[0],:]
        C_mean[i,:] = np.mean(temp,axis=0)
        S_W[i,:,:] = np.dot((temp-repmat(C_mean[i,:],temp.shape[0],1)).transpose(),(temp-repmat(C_mean[i,:],temp.shape[0],1)))
        #S_B = S_B + np.dot((m_class - m_total).transpose(),(m_class-m_total))
        #S_W = S_W + np.dot((temp - repmat(m_class,temp.shape[0],1)).transpose(),(temp - repmat(m_class,temp.shape[0],1)))
    k = 0
    for i in range(classes):
        for j in range(i+1,classes,1):
            weight[:,k] = np.dot(inv(S_W[i,:,:] + S_W[j,:,:]), (C_mean[i,:]-C_mean[j,:]).transpose())
            k = k + 1
    return weight