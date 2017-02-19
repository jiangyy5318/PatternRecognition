from numpy import *
import numpy as np
import scipy.io as sio
from numpy.matlib import repmat

def sigmoid(theta,x):
    return 1/(1+np.exp(-np.dot(x,theta)))


def LogisticRegression(X,Y,theta,Iter,k = 4,Lambda = 1e-4):
    N = X.shape[0]
    for i in range(Iter):
        #Codes below applied stochastic gradient decent into logistic regression.
        permutation = np.random.permutation(N)[0:k]
        tempX = X[permutation,:]
        tempY = Y[permutation,:]

        grad = np.dot(tempX.T,(tempY-sigmoid(theta,tempX)))
        theta = theta + Lambda*grad
    return theta

def Accuracy(X,Y,theta):
    return float(np.sum(np.abs(sigmoid(theta,X)-Y)< 0.5))/X.shape[0]

if __name__ == "__main__":
    N = 200
    mat = sio.loadmat('..\mnist.mat')
    chooseTrain = np.where(np.array(mat['train_Y'][0]) < 2)
    chooseTest = np.where(np.array(mat['test_Y'][0]) < 2)
    train_X = mat['train_X'][chooseTrain,:][0]
    train_Y = mat['train_Y'][0,chooseTrain].transpose()
    test_X = mat['test_X'][chooseTest,:][0]
    tets_Y = mat['test_Y'][0,chooseTest].transpose()
    train_X = np.concatenate((train_X,np.ones((train_X.shape[0],1))),axis=1)
    test_X = np.concatenate((test_X, np.ones((test_X.shape[0], 1))), axis=1)
    theta = np.ones((train_X.shape[1],1),dtype=float)/train_X.shape[1]
    Lambda = 1e-4
    Iter = 40
    batch = 20
    theta = LogisticRegression(train_X,train_Y,theta,40,k = batch,Lambda = 1e-4)
    accu = Accuracy(train_X,train_Y,theta)
    print 'Accuracy = %f,Iterations = %d,batch = %d'%(accu,Iter,batch)
    #print theta