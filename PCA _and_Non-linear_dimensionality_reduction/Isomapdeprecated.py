import matplotlib.pyplot as plt
from numpy import *
from numpy.linalg import *
import random
import numpy as np
import fpconst
import matplotlib.cm as cm
#from sklearn import manifold
#from manifold import MDS
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigsh

def shortest_paths(adj):
    (n,m) = adj.shape
    assert n == m
    for k in range(n):
        adj = np.minimum( adj, np.add.outer(adj[:,k],adj[k,:]) )
    return adj
#
"""
def mds(D, dimensions = 2):
    #E = D + np.transpose(D)
    Dims = D.shape[0]
    l = np.ones((Dims,1))
    J = np.eye(Dims) - l*np.transpose(l)/Dims
    B = -0.5*J*D*J
    #w, v = LA.eig(B)
    #w,v = eigsh(B, 2, which='LM')
    #print w
    #diagw = np.diag(w)**0.5
    #X = np.dot(v,diagw)

    #print "="*80
    #print w,v
    #diagw = np.diag(w)**0.5
    #X = np.dot(v,diagw)
    #ArrayX = np.array(X)
    #ccc = np.sort(w)
    #print "="*60
    #print ccc
    #X = np.dot(v[:,:2],np.diag(np.sqrt(w[:2])))


    return X
"""
def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.

    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n

    # YY^T
    B = -H.dot(D**2).dot(H)/2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals


def mds(d, dimensions = 2):

    (n,n) = d.shape
    E = (-0.5 * d**2)

    # Use mat to get column and row means to act as column and row means.
    Er = mat(mean(E,1))
    Es = mat(mean(E,0))

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = array(E - transpose(Er) - Es + mean(E))

    [U, S, V] = svd(F)
    Y = U * np.sqrt(S)
    return Y[:,0:dimensions]

def norm(vec):
    return np.sqrt(np.sum(vec**2))

#def flatidx(idx):
#    adds = np.arange(0, len(idx) ** 2, len(idx))
#    return np.ravel(np.transpose(idx) + adds)
"""
def MakeGraph(Graph,k):
    #KDistance = Graph.copy()
    #KD = np.sort(np.array(KDistance),axis=0)
    #print KD
    n = Graph.shape[0]
    #print np.diag(np.ones(n))
    Graph = Graph + np.diag(ones(n)*fpconst.PosInf)
    print Graph
    AfterSort = np.sort(Graph,axis=0)
    print Graph
    Temp = np.repeat(AfterSort[k],n,axis=0)
    print Temp
    ld = np.where(np.less(Graph, Temp), Graph, fpconst.PosInf)
    print ld
    #Index = np.argsort(Graph,axis=0)
    #print Index
    # = np.matrix(KD[k])
    #print MatrixKD
    #(m,n) = Graph.shape
    #Temp = np.repeat(MatrixKD,n,axis=0)
    #ld = np.where(np.less(Graph, Temp), Graph, fpconst.PosInf)
    ld = np.minimum(ld, np.transpose(ld))
    print ld
    #return ld
    return ld
"""
def MakeGraph(Graph,k):
    #KDistance = Graph.copy()+
    n = Graph.shape[0]
    Graph = Graph + np.diag(ones(n)*fpconst.PosInf)
    print '='*10,1
    print Graph
    KD = np.sort(np.array(Graph),axis=0)
    print '='*10,2
    print KD
    MatrixKD = np.matrix(KD[k])
    print '='*10,3
    print MatrixKD
    Temp = np.repeat(MatrixKD,n,axis=0)
    print '='*10,4
    print Temp
    ld = np.where(np.less(Graph, Temp), Graph, fpconst.PosInf)
    print '='*10,5
    print ld
    ld = np.minimum(ld, np.transpose(ld))
    #print '='*10,6
    print ld
    return ld

def Generate3shapePoint():
    x = random.uniform(-pi,pi)
    z = random.uniform(0,2*pi)
    y = 2*sin(x)
    return np.array([x,y,z], dtype=float)




if __name__ == "__main__":
    N = 1000
    points = np.zeros((N,3),dtype=float)
    #Generate points
    for i in range(N):
        points[i] = Generate3shapePoint()

    distance = np.zeros((N,N))
    for (i, pointi) in enumerate(points):
        for (j, pointj) in enumerate(points):
            distance[i,j] = norm(pointi - pointj)

    adj = MakeGraph(distance,9)
    #print adj

    Shoretestadj = shortest_paths(adj)
    Y = mds(Shoretestadj)
    #mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    #Y = mds.fit(Shoretestadj)
    #print Y
    #print Shoretestadj
    #print ArrayX
    colors = cm.rainbow(np.linspace(0,1,4))

    #for x,c in zip(Y):
    for i in range(len(Y)):
        plt.scatter(Y[i,0],Y[i,1],color=colors[int(points[i,1]/1.58)+2])
    plt.legend()
    plt.show()
    #pylab.figure(1)
    #plt.plot(Y[:,0],Y[:,1],'.')
    #plt.show()
    #Plot
    #colors = cm.rainbow(np.linspace(0,1,len(points)))
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    for i in range(len(Y)):
        ax3D.scatter(points[i,0],points[i,1],points[i,2], c=colors[int(points[i,1]/1.58)+2], marker='o')
    plt.show()
    print 'end'