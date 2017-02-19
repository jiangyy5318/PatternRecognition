#!/usr/bin/python
#-*- coding: UTF-8 -*-
from numpy import *
import numpy as np
from numpy.linalg import inv
import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('least_sq.mat')
data = mat['train_small']
train_X = data['X'][0,0]
train_Y = data['y'][0,0]
test = mat['test']
test_X = test['X'][0,0]
test_Y = test['y'][0,0]
#transpose()

Lambda = range(0.01,2.001,0.01) #a series of L1-norm penalty
W_0 = inv(train_X.transpose().dot(train_X)).dot(train_X.transpose().dot(train_Y))

# step 2 Train weight vectors with different penalty constants
W = least_sq_multi(train_X,train_Y,Lambda,W_0)




