#!/usr/bin/python
#-*- coding: UTF-8 -*-
import numpy as np
import scipy.io as sio
from adaboost import adaboost

N = 200
mat = sio.loadmat('../mnist.mat')

print mat['test_X'].shape[0]
chooseTrain = np.random.permutation(mat['train_X'].shape[0])
chooseTest = np.random.permutation(mat['test_X'].shape[0])
maxIter = 20
train_X = mat['train_X']
train_Y = mat['train_Y']
test_X = mat['test_X']
test_Y = mat['test_Y']
#print train_X.shape
#print train_Y.shape
#print chooseTrain.shape
e_train,e_test,maxIter = adaboost(train_X[chooseTrain[0:10000],:],train_Y[0,chooseTrain[0:10000]],test_X[chooseTest[0:1000],:],test_Y[0,chooseTest[0:1000]],maxIter)
#print e_train
#print e_test

import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = range(maxIter)

# red dashes, blue squares and green triangles
plt.plot(t, e_train, 'b:', t,e_test , 'k-', t, e_train[0]*np.ones(maxIter), 'b--',e_test[0]*np.ones(maxIter),'b--')
plt.show()
#plot

