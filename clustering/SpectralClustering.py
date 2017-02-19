
from numpy.linalg import *
from KMeans import *
from choosedata import choose
from nmi import IndexIMI

def getAdjacency(data,k):
    return MakeGraph(cdist(data, data, 'euclidean'),k)
    """
    N = len(data)
    distance = np.zeros((N,N))
    for (i, pointi) in enumerate(data):
        for (j, pointj) in enumerate(data):
            distance[i,j] = norm(pointi - pointj)
    return MakeGraph(distance,k)
    """

def MakeGraph(Graph,k):
    #maybe npargsort Graph[KD[0:k,:]  = 1,the rest = 0 works better.
    KD = np.sort(np.array(Graph),axis=0)
    Temp = np.repeat(np.matrix(KD[k]),KD.shape[0],axis=0)
    ld = np.where(np.less(Graph, Temp), 1, 0)
    ld = np.where(np.add(ld,ld.transpose()),1,0)
    return ld
    #"""

def getNormLaplacian(W):
    D=np.diag([np.sum(row) for row in W])
    L=D-W
    Dn=np.power(np.linalg.matrix_power(D,-1),0.5)
    Lbar=np.dot(np.dot(Dn,L),Dn)
    return Lbar

def norm(vec):
    return np.sum(vec**2)


def spectral_clustering(L,k):
    [U, S, V] = svd(L)
    return U[:,0:k]

"""
def RandomIntialKPoints(K,data):
    Temp = np.zeros((K,data.shape[1]),dtype=float)
    for i in range(K):
        Temp[i] = data[random.randint(0,len(data)-1)]
    return Temp
"""


if __name__ == '__main__':
    N = 200
    mat = sio.loadmat('mnist.mat')
    test_X,test_Y = choose(np.array(mat['test_X']),np.array(mat['test_Y'])[0,],N)

    #Compared to kmeans,spectural clustering reduced dimensions before clustering
    Laplaceian = getNormLaplacian(getAdjacency(test_X,5))
    reduced =  spectral_clustering(Laplaceian,3)

    #Codes below keep same with kmeans
    K = 10
    Iteration = 100
    start_time = time.time()
    Centers = RandomIntialKPoints(K,test_X)
    for i in range(Iteration):
        pred = AssignCenter(test_X,Centers)
        Centers = updateCenters(test_X,pred,Centers)
    NMI = IndexIMI(test_Y,pred)
    print NMI
    elapsed_time = time.time() - start_time
    print 'Elapsed time:',elapsed_time,'seconds'


