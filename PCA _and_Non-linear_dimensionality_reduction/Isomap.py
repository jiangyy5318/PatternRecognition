import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.distance import cdist
import random
from mds import *
import matplotlib.cm as cm
def knn(X,K):
    D = cdist(X, X, 'euclidean')
    KD = np.sort(np.array(D), axis=1)
    radis = np.repeat(KD[:, K].reshape(KD.shape[0], 1), KD.shape[1], axis=1)
    ld = np.where(np.less(D, radis), D, 0)
    return  np.minimum(ld,np.transpose(ld))

def shortest_paths(adj):
    (n,m) = adj.shape
    assert n == m
    for k in range(n):
        adj = np.minimum( adj, np.add.outer(adj[:,k],adj[k,:]) )
    return adj

def Generate3shapePoint(N,M):
    data = np.zeros((N,4), dtype=float)
    for i in range(N):
        data[i,0] = random.uniform(-math.pi,math.pi)
        data[i,2] = random.uniform(0,2*math.pi)
        data[i,1] = math.fabs(data[i,0])
        data[i,3] = int((data[i,0]+4)*M/8)
    return data

if __name__ == "__main__":
    N = 1000
    M = 30
    colors = cm.rainbow(np.linspace(0, 1, M + 1))
    #print colors
    data = Generate3shapePoint(N,M)
    geodistance = knn(data[:,0:3],M)

    X,temp = mds(geodistance,dimensions=2)
    ArrayX = np.array(X)
    ArrayColor = np.array(data[:,3])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x, c in zip(data, ArrayColor):
        ax.scatter(x[0], x[1], x[2], c=colors[c], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


    for x, c in zip(ArrayX, ArrayColor):
        plt.scatter(x[0], x[1], c=colors[c], marker='o',label='')#, color=colors[c])
    plt.legend()
    plt.title('isomap locations')
    plt.show()

