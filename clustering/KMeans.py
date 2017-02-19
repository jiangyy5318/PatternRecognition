from numpy import *
import numpy as np
import time
import scipy.io as sio
from scipy.spatial.distance import *
from nmi import IndexIMI
from choosedata import choose

#time complexity O(n),only for this problem

#only use matrix implement kmeans.
def RandomIntialKPoints(K,X):
    num = X.shape[0]
    index = np.random.permutation(num)[0:K]
    return X[index,]

#Assign all the dataset into different clusterings
def AssignCenter(X,Centers):
    dis = cdist(X, Centers, 'euclidean')
    return argmin(dis,axis=1)

#Update new centers
def updateCenters(X,pred,Centers):
    tempCenters = np.zeros(Centers.shape)
    for i in range(Centers.shape[0]):
        select = np.array(np.where(pred == i))[0,]
        tempCenters[i,] = mean(X[select,],axis=0)
    return tempCenters

if __name__ == '__main__':

    N = 200
    mat = sio.loadmat('mnist.mat')
    test_X,test_Y = choose(np.array(mat['test_X']),np.array(mat['test_Y'])[0,],N)
    K = 10
    Iteration = 20
    start_time = time.time()
    Centers = RandomIntialKPoints(K,test_X)
    for i in range(Iteration):
        pred = AssignCenter(test_X,Centers)
        Centers = updateCenters(test_X,pred,Centers)
    NMI = IndexIMI(test_Y,pred)
    print NMI
    elapsed_time = time.time() - start_time
    print 'Elapsed time:',elapsed_time,'seconds'

    #print 'end'