import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

class bpnn:
    def __init__(self,data,label,test_data,test_label):
        self.data = np.concatenate((data,np.ones((data.shape[0],1),dtype=float)),axis= 1)
        C = len(np.unique(label))
        self.label = np.zeros((self.data.shape[0],C),dtype=float)
        for i in range(self.label.shape[0]):
            self.label[i,label[i]] = 1
        self.theta = 2*np.random.rand(self.data.shape[1],C) - 1
        self.test_data = np.concatenate((test_data,np.ones((test_data.shape[0],1),dtype=float)),axis= 1)
        self.test_label = test_label


    def Train(self,maxIter,testIter):
        self.errorrate = np.zeros((2,maxIter/testIter))
        for i in range(maxIter):
            perm = np.random.permutation(self.data.shape[0])
            Z2 = self.data[perm[0:100],:].dot(self.theta)
            a2 = 1 / (1 + np.exp(-Z2))
            Delta2 = -(self.label[perm[0:100],:] - a2) * a2 * (1- a2)
            grad = self.data[perm[0:100],:].T.dot(Delta2)
            self.theta = self.theta -  0.01*grad
            if i % testIter == 0:
                self.errorrate[0,i/testIter] = i
                self.errorrate[1,i/testIter] = self.Errorratecalc()

    def Predict(self,test):
        Temp = np.concatenate((test,np.ones((test.shape[0],1),dtype=float)),axis= 1)
        Z2 = np.dot(Temp,self.theta)
        a2 = 1 / (1 + np.exp(-Z2))
        pred = a2.argmax(axis = 1)
        return pred
    def Errorratecalc(self):
        Z2 = np.dot(self.test_data,self.theta)
        a2 = 1 / (1 + np.exp(-Z2))
        pred = a2.argmax(axis = 1)
        return float(sum(pred != self.test_label)) / self.test_label.shape[0]


if __name__ == "__main__":
    mat = sio.loadmat('fc6.mat')
    train_X = mat['train_X']
    train_Y = mat['train_Y'][0]
    test_X = mat['test_X']
    test_Y = mat['test_Y'][0]

    start = time.time()
    bpnnfc6 = bpnn(train_X,train_Y,test_X,test_Y)
    bpnnfc6.Train(20000,10)
    print 'fc6 error rate = ',float(sum(bpnnfc6.Predict(test_X) != test_Y)) / test_Y.shape[0]
    print 'fc6 bpnn Elapsed time',time.time() - start,'seconds'

    mat = sio.loadmat('pool5.mat')
    train_X = mat['train_X']
    train_Y = mat['train_Y'][0]
    test_X = mat['test_X']
    test_Y = mat['test_Y'][0]

    start = time.time()
    bpnnpool5 = bpnn(train_X,train_Y,test_X,test_Y)
    bpnnpool5.Train(20000,10)
    print 'pool5 error rate = ',float(sum(bpnnpool5.Predict(test_X) != test_Y)) / test_Y.shape[0]
    print 'pool5 bpnn Elapsed time',time.time() - start,'seconds'

    plt.plot(bpnnfc6.errorrate[0], bpnnfc6.errorrate[1],label = 'fc6')
    plt.plot(bpnnpool5.errorrate[0],bpnnpool5.errorrate[1],label = 'pool5')
    plt.xlabel(r"Iter",fontsize=20)
    plt.ylabel(r'Error rate')
    plt.legend(loc = 'upper right')
    plt.title('Error rate vs iter ')
    plt.show()


