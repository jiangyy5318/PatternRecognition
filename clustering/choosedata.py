import numpy as np
import scipy.io as sio

def choose(X,Y,K = 20):
    #print Y.shape
    length = X.shape[0]
    index = np.random.permutation(length)[0:K]
    return X[index,],Y[index]

if __name__ == '__main__':
    mat = sio.loadmat('mnist.mat')
    test_X,test_Y = choose(np.array(mat['test_X']),np.array(mat['test_Y'])[0,],200)
    print test_X.shape
    print test_Y

