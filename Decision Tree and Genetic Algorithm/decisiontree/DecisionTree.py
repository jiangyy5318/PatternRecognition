#!/usr/bin/python
#-*- coding: UTF-8 -*-
from numpy import *
import numpy as np
import scipy.io as sio

class Node(object):
    def __init__(self,data,leaf):
        self.data = data
        self.leaf = leaf

    def IsLeaf(self):
        return self.leaf != 0

    def add_LeftChild(self,obj):
        self.leftChild = obj

    def add_RightChild(self,obj):
        self.rightChild = obj


def DividedOriginalDataIntoTrainAndTest(OriginalX,OriginalY):
    ccc = np.concatenate((OriginalX,OriginalY),axis=1)
    (m,n) = ccc.shape
    Train = []
    Test = []
    for i in range(m):
        if(i%10 == 9):
            Test.append(ccc[i])
        else:
            Train.append(ccc[i])
    return Train,Test

#Vecorize InfoGain
def SearchLargestInfoGain(X,Width):

    sheet = np.zeros((1200,2,9),dtype=int)
    sheetBase = np.zeros((9),dtype=int)
    (m,n,l) = sheet.shape
    #st = random.randint(0,1200)
    for e in X:
        sheetBase[e[1200]-1] = sheetBase[e[1200]-1] + 1
        for i in range(1200):
            sheet[i,e[i],e[1200]-1] = sheet[i,e[i],e[1200]-1] + 1
    sheet = sheet + 1
    #print sheet
    Temp = np.array(sheetBase,dtype=float)/ sheetBase.sum()
    #print Temp
    sheetsumaxis2 = sheet.sum(axis=2)
    #print sheetsumaxis2
    sheetsumaxis12 = sheetsumaxis2.sum(axis = 1)
    #print sheetsumaxis12
    Ipn = np.array(sheet,dtype=float) / sheetsumaxis2.reshape((m,n,1)).repeat(l,axis = 2)
    #print Ipn
    Percent = np.array(sheetsumaxis2,dtype=float) / sheetsumaxis12.reshape((m,1)).repeat(n,axis = 1)
    #print Percent
    NewEntropy = ((-np.log2(Ipn) * Ipn).sum(axis=2) * Percent).sum(axis=1)
    #print NewEntropy
    return  np.argmax((-np.log2(Temp)*Temp).sum() - NewEntropy)



def splitTree(X,Index):

    Left = []
    Right = []
    j = 0
    k = 0
    print Index
    for e in X:
        if e[Index] == 0:
            Left.append(e)
        else:
            Right.append(e)
    return Left,Right

def StopCondition(X):
    C = []
    for e in X:
        C.append(e[1200])
    if(len(unique(C)) == 1):
        return C[0]
    else:
        return 0

def Majority(X):
    Cnt = np.zeros(9,dtype=int)
    for e in X:
        Cnt[e[1200]-1] = Cnt[e[1200]-1] + 1
    return np.argmax(Cnt) + 1

def Recursive(X):

    Condition = StopCondition(X)
    if(Condition != 0):
        return Node(1500,Condition)
    index =  SearchLargestInfoGain(X,120)
    p = Node(index,0)
    LeftX,RightX = splitTree(X,index)

    if len(LeftX) == 0 or len(RightX) == 0:
        return Node(1500,Majority(X))
    del X
    m = Recursive(LeftX)
    n = Recursive(RightX)
    p.add_LeftChild(m)
    p.add_RightChild(n)
    return p

def SearchIndexTree(p,test):
    if p.IsLeaf() == False:
        if test[p.data] == 0:
            p = p.leftChild
        else:
            p = p.rightChild
        return SearchIndexTree(p,test)
    return p.leaf

if __name__ == '__main__':
    mat = sio.loadmat('Sogou_webpage.mat')
    Train,Test = DividedOriginalDataIntoTrainAndTest(mat['wordMat'],mat['doclabel'])
    import time
    start = time.time()
    P = Recursive(Train)
    TrainTime = time.time() - start
    start = time.time()
    Predict = []
    for e in Test:
        s = SearchIndexTree(P,e)
        Predict.append(s)
    Accuracy = float(sum([e == f[1200]  for e,f in zip(Predict,Test)]))/len(Predict)
    TestTime = time.time() - start
    print 'Accuracy',Accuracy,'Train elapsed time:',TrainTime,'Test elapsed time',TestTime