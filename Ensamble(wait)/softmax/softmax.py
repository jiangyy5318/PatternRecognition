from numpy import *
import numpy as np
import scipy.io as sio
from numpy.matlib import repmat
"""
    \[
    \nabla_{\theta_j}J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[x^{(i)}(1\left\lbrace y^{(i)}=j\right\rbrace - p(y^{(i)}=j|x^{(i)};\theta)\right] + \lambda\theta_j
    \]
"""
def h_theate(X,theta):
    h1 = np.exp(np.dot(X,theta))
    h2 = repmat(np.sum(h1,axis=0),h1.shape[0],1)
    return np.divide(h1,h2)
#stochastic gradient descent.
def softmax_predict(X,theta):
    return np.argmax(h_theate(X,theta),axis=1)

def softmax_cost(X,label,theta):
    loghtheta = np.log(h_theate(X,theta))
    N = X.shape[0]
    #indicator = label
    Jtheta = -np.sum(label*loghtheta)/N
    return

def softmax_train(X,Y,theta,g=1e-4,Lambda = 1e-4,Iter = 500,batch = 20):

    label = np.zeros((train_Y.shape[0],C))
    N = X.shape[0]
    for i in range(N):
        label[i,train_Y[i]] = 1

    for i in range(Iter):
        permutation = np.random.permutation(N)[0:batch]
        tempX = X[permutation, :]
        tempY = label[permutation, :]
        #print tempY.shape
        prob = h_theate(tempX,theta)
        #print prob.shape
        grad = -np.transpose(tempX).dot(tempY-prob)/tempX.shape[0] + theta * Lambda
        theta = theta - g*grad
    #print theta
    cost = softmax_cost(X,label,theta)

    return theta,cost

if __name__ == "__main__":
    mat = sio.loadmat('..\mnist.mat')

    # before entering softmax,extend X with a column of 1 as the bias
    train_X = np.concatenate((mat['train_X']/255, np.ones((mat['train_X'].shape[0],1))), axis=1)
    train_Y =mat['train_Y'][0]
    test_X = np.concatenate((mat['test_X']/255, np.ones((mat['test_X'].shape[0],1))), axis=1)
    test_Y = mat['test_Y'][0]

    #intialize theta with 1/m
    C = np.unique(train_Y).shape[0]
    #print C
    #Codes below requre train_Y \in 0:C
    theta = np.ones((train_X.shape[1],C),dtype=np.float64)/train_X.shape[1]/20
    #theta = 0.005 * np.random.randn(train_X.shape[1],C)
    g = 1e-5
    Lambda = 1e-5
    Iter = 1000
    batch = 50
    theta,cost = softmax_train(train_X,train_Y,theta,g=g,Lambda = Lambda,Iter = Iter,batch = batch)
    #permutation = np.random.permutation(test_X.shape[0])[0:100]
    #print softmax_predict(test_X[permutation,],theta)
    accu = float(np.sum(softmax_predict(test_X,theta) == test_Y))/test_X.shape[0]
    print "Accuracy = %f,Iterations = %d,batch = %d" % (accu, Iter, batch)
    """
    compared with others,Accuracy of mine falls behind.parameter problem.
    """
