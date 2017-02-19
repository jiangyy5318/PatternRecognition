import sys
import numpy as np
from numpy import *
import csv
from datetime import *
import copy
import random

dir = 'data/'
def load(fType):
    trainData   = dir + fType + ".data"
    trainLabel  = dir + fType + ".label"
    data = fromfile(trainData, dtype=int, sep='\n')
    label = fromfile(trainLabel, dtype=int, sep='\n')
    data.shape = (-1,3)
    label.shape = (-1,1)
    return data,label

def ExtractFeature(data, m):
    articles        = unique(data[:,0])
    cols            = articles.shape[0]    #Get the number of articles
    feature         = matrix(zeros([m+1, cols]))

    feature[m, :] = 1; # for the lnP(Ci)
    for [articleId, wordId, count] in data:
        if wordId <= m:
            feature[wordId-1, articleId-1] += count

    return feature

def chooseTwoLabelData(data,label):
    choose = np.where(label<2)
    print choose
    return data[choose,:][0].label[choose][0]


c = 20
(train, label) = load('train')
(test, tLabel) = load('test')

print 'Load Data Over'

m = max(train[:, 1])
trainFeature = ExtractFeature(train, m)
testFeature = ExtractFeature(test, m)

print chooseTwoLabelData(trainFeature,label)
