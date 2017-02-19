import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import *
from numpy.linalg import *
from numpy.random import *

def Normalize(v):
    Sum = math.sqrt(sum(v[i]*v[i] for i in range(len(v))))
    return 1/Sum*v

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

    [U, S, V] = svd(F)

    Y = U * sqrt(S)

    return (Y[:,0:dimensions], S)



if __name__ == "__main__":
    Ratednum = np.zeros((20,5))
    #for each_line in fileinput.input(input_file):
    #do_something(each_line)
    for line in open('u.data'):
        Rated = map(int,line.replace('\n','').split('\t'))
        if(Rated[1] > 20):
            continue
        Ratednum[Rated[1]- 1,Rated[2]-1] =  Ratednum[Rated[1]- 1,Rated[2]-1]+1
    #for i in range(len(Ratednum)):
    #    Ratednum[i] = Normalize(Ratednum[i])
    Temp = [Normalize(v) for v in Ratednum]
    #print Tem
    #print Ratednum
    RatedMatrix = np.matrix(Temp)
    Similarity = np.dot(RatedMatrix,np.transpose(RatedMatrix))
    ArrayX,eig = mds(Similarity,2)
    #print ArrayX
    colors = cm.rainbow(np.linspace(0,1,len(ArrayX)))

    #for x,c in zip(ArrayX,colors):
    for i in range(len(ArrayX)):
        plt.scatter(ArrayX[i,0],ArrayX[i,1],label=str(i+1),color=colors[i])
    plt.legend()
    plt.show()
    #pylab.figure(1)
    #pylab.plot(ArrayX[:,0],ArrayX[:,1],'.',Color)

    #pylab.figure(2)
    #pylab.plot(points[:,0], points[:,1], '.')

    #pylab.show()


