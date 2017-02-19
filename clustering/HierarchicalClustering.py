from numpy import *
import numpy as np
import scipy.io as sio
import fpconst
import time
import sys
from nmi import IndexIMI
from choosedata import choose


def norm(vec):
    return np.sum(vec**2)

def Singlelinkage(Group1,Group2):
    Min = sys.maxint
    for e in Group1:
        for f in Group2:
            Temp = norm(e-f)
            if(Temp < Min):
                Min = Temp
    return Min

def Completelinkage(Group1,Group2):
    Max = 0
    for e in Group1:
        for f in Group2:
            Temp = norm(e-f)
            if(Temp > Max):
                Max = Temp
    return Max

def MinNorm(Group1,Group2):
    Center1 = mean(Group1,axis=0)
    Center2 = mean(Group2,axis=0)
    return norm(Center1-Center2)

#def IntialData(data):
def InitialDistanceMatrix(data):
    n = len(data)
    D = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(i+1,n,1):
            D[i,j] = MinNorm(data[i],data[j])
    return D + np.transpose(D) + np.diag(np.ones(n,dtype=float)*fpconst.PosInf)

def Combine2Group(min,max,data,Groups,matrix):
    data[min] = concatenate((data[min],data[max]),axis=0)
    #Groups[min]= concatenate((Groups[min],Groups[max]),axis=1)
    Groups[min].extend(Groups[max])
    del data[max]
    del Groups[max]
    matrix = np.delete(matrix,max,axis=0)
    matrix = np.delete(matrix,max,axis=1)
    n = len(data)
    UpdateDistance = np.zeros((n,1),dtype=float)
    for i in range(n):
        if(i == min):
            UpdateDistance[i,0] = fpconst.PosInf
            continue
        UpdateDistance[i,0] = Completelinkage(data[i],data[min])
    matrix[:,min] = UpdateDistance[:,0]
    matrix[min,:] = np.transpose(UpdateDistance)[0,:]
    return data,Groups,matrix

def GetargMinMatrix(matrix):
    index = np.argmin(matrix)
    #print index,matrix.shape
    return int(index/len(matrix)),index%len(matrix)

if __name__ == '__main__':

    N = 200
    mat = sio.loadmat('mnist.mat')
    test_X,test_Y = choose(np.array(mat['test_X']),np.array(mat['test_Y'])[0,],N)
    #test_X = np.array([[1,1],[1,2],[4,1],[4,3]])
    start_time = time.time()
    Groups = []
    data = []
    for i in range(len(test_X)):
        data.append(np.reshape(test_X[i],(1,len(test_X[i]))))
        Groups.append([int(test_Y[i])])

    matrix = InitialDistanceMatrix(data)
    K = 10
    while(len(data) > K):
        c1,c2 = GetargMinMatrix(matrix)
        #print c1,c2
        #print matrix
        data,Groups,matrix = Combine2Group(min(c1,c2),max(c1,c2),data,Groups,matrix)
    #print Groups
    Predict = []
    Standard = []
    for i, val in enumerate(Groups):
        for j,ccc in enumerate(val):
            Predict.append(i)
            Standard.append(ccc)
    #print Predict,Standard

    NMI = IndexIMI(Predict, Standard)
    print NMI
    elapsed_time = time.time() - start_time
    print 'Elapsed time:',elapsed_time,'seconds'
