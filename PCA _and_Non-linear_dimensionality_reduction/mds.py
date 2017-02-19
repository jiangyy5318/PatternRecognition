import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.linalg import *
from numpy import *

def mds(D):
    """
    :param Distance: input np.matrix
    :return:
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones(D.shape)/n
    B = -0.5*J*D*J
    w,v = eigsh(B, k=2 , which='LM')
    diagw = np.diag(w)**0.5
    X = np.dot(v,diagw)
    return X


def mdssvd(D,dimensions = 2):
    n = D.shape[0]
    J = np.eye(n) - np.ones(D.shape)/n
    B = -0.5*J*D*J
    [U, S, V] = svd(B)
    Y = U * np.sqrt(S)
    return Y[:, 0:dimensions]


def mds(d, dimensions = 2):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """
    (n,n) = d.shape
    E = (-0.5 * d**2)
    # Use mat to get column and row means to act as column and row means.
    Er = mat(mean(E,1))
    Es = mat(mean(E,0))
    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = array(E - transpose(Er) - Es + mean(E))
    #
    # XX^T = -\frac{1}{2}JDJ =
    # JDJ = (I - 11^T/n)D(I - 11^T/n) = (D -  1*mean(D,1)(I - 11^T/n) = D - repmat(mean(D,1),1,n) - repmat(mean(D,0),n,1) + repmat(mean(E),n,n)

    [U, S, V] = svd(F)

    Y = U * sqrt(S)

    return (Y[:,0:dimensions], S)
