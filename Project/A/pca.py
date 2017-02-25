import numpy as np
from numpy.linalg import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot
from pylab import *
import random
import pandas as pd

class pca:
    def __init__(self,data,test):
        self.data = data
        self.test = test
    def mean(self):
        mean = self.data.mean(axis = 0)
        mean = mean.reshape(1,mean.shape[0])
        return mean
    #def Corvariance(self):
    def xRot(self,k):
        mean = self.mean()
        X = self.data - repeat(mean,self.data.shape[0],axis=0)
        CoVariance = np.dot(np.transpose(X),X)/X.shape[0]
        [U, S, V] = svd(CoVariance)
        if sum(S > 0) < k:
            k = sum(S>0)
        return np.dot(X,U[:,0:k]),np.dot(self.test - repeat(mean,self.test.shape[0],axis=0),U[:,0:k])

    #def xRot(self,k):
    #    self.Rot(k)
    #    rot = np.dot(X,U[:,0:k])
    #    return rot
"""
def loaddata(datafile):
    return np.array(pd.read_csv(datafile,sep="\t",header=-1)).astype(np.float)

def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0])
        axis_y2.append(dataArr2[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()
"""

#if __name__ == '__main__':
    #mat = sio.loadmat('mnist.mat')
    #datafile = "data.txt"
    #XMat = loaddata(datafile)
    #pca = pca(XMat)
    #xRot = pca.xRot(2)
    #plotBestFit(xRot,XMat)
    #print xRot



#def featureExtract(X,Y):
#    Classes = len(unique(Y))
#
#    return 0