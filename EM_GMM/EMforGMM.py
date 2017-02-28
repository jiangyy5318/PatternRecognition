import numpy as np
from numpy.linalg import *
from math import *

mu1 = np.array([[0.0],
                [0.0],
                [0.0]])
mu2 = np.array([[1.0]
                [1.0]
                [1.0]])
mu3 = np.array([[5.0]
                [5.0]
                [5.0]])
sigma1 = sigma12 = sigma13 = np.array([[1, 0, 0]
                                       [0, 1, 0]
                                       [0, 0, 1]])
pw1 = 0.1
pw2 = 0.3
pw3 = 0.6

def gaussian(x,mu,sigma):
    #x,mu:shape(3,1)
    #sigma (3,3)
    sigmainv = inv(sigma)
    d = x.shape[0]
    m = np.dot(np.transpose(x-mu),np.dot(sigmainv,x-mu))
    sigmadet = det(sigma)
    return 1 / sqrt(pow(2*pi,d)) * exp(-m/2)

#def GMM(x):
    #p =

#def generatedata():


if __name__ == "__main__":

    #Generat Data:
    #
    print
