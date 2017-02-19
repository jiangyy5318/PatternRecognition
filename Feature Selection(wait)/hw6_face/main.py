#!/usr/bin/python
#-*- coding: UTF-8 -*-
from numpy import *
import numpy as np
import scipy.io as sio
from MultipleClasses import MultipleClasses
from oneversusone import oneversusone
from knn import knn
import matplotlib.pyplot as plt

mat = sio.loadmat('orl_faces.mat')
data = mat['data']
label = mat['label']

max = 400#10's multiples
#the 6 below codes split data into train and test
choose = np.zeros((400), dtype=bool)
choose[np.arange(0,400,5)] = True
train_X = data[np.where( choose == False )[0],:]
train_Y = label[np.where( choose == False )[0],0]
test_X = data[np.where( choose == True )[0],:]
test_Y = label[np.where( choose == True )[0],0]

if True:
    #C(C-1)/2 features.
    #the one-versus-one strategy may lead to some ambiguous results, you can illustrate this fact with some simple hand-drawn images
    weight = oneversusone(train_X,train_Y,d=20)
    succ = knn(train_X.dot(weight),train_Y,test_X.dot(weight),test_Y,3,metric='euclidean')
if False:
    weight = MultipleClasses(train_X,train_Y,d=20)
    succ = knn(train_X.dot(weight),train_Y,test_X.dot(weight),test_Y,3,metric='euclidean')

if False:
    #show that accuracy vary with the number of feature
    weight = MultipleClasses(train_X,train_Y,d=100)
    list = []
    for i in range(1,100,1):
        succ = knn(train_X.dot(weight[:,0:i]),train_Y,test_X.dot(weight[:,0:i]),test_Y,3,metric='euclidean')
        list.append(succ)
    plt.plot(range(len(list)), list, 'b:')
    plt.show()
