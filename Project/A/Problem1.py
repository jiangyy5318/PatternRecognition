import numpy as np
from numpy import *
import os
from PIL import Image
from Knn import *
from pca import *
from numpy.linalg import *
import scipy.io as sio
#from Softmax import *

#from nerualnetwork import *

#Generate data
def GeneratedataSet(A):
    Cnt = sum([len(filenames) for parent,dirnames,filenames in os.walk(A)])
    print Cnt
    X = np.zeros((Cnt,100,100),dtype= float)
    i = 0
    Y = np.zeros((1,Cnt),dtype= int)
    for root,dirs,files in  os.walk(A):
        for f in files:
            im = np.array(Image.open(os.path.join(root,f)).resize((100,100),Image.BILINEAR).convert('L'),'f')
            X[i,:,:] = im
            Y[0,i] = int(os.path.split(root)[-1][1])
            i = i + 1
    return X.reshape(Cnt,-1),Y


if __name__ == '__main__':

    train_X,train_Y = GeneratedataSet('../train')
    test_X,test_Y = GeneratedataSet('../val')
    sio.savemat('Temp.mat',mdict={'train_X':train_X,'train_Y':train_Y,'test_X':test_X,'test_Y':test_Y})
    mat = sio.loadmat('Temp.mat')
    Train_xRot = mat['train_X']
    train_Y = mat['train_Y']
    Test_xRot = mat['test_X']
    test_Y = mat['test_Y']



    ccc = knn(Train_xRot,train_Y[0])
    pred = ccc.KnnPredict(Test_xRot,1,'euclidean')
    print 'knn error :',ccc.ErrorRate(pred,test_Y[0])

    """
    softmax = softmax(Train_xRot,train_Y,0.0001)
    softmax.Train(10000)
    #TrainTime = time.time() - start
    #print 'Elapsed Train time',TrainTime
    print 'softmax error rate:',softmax.ErrorRate(Test_xRot,test_Y)
    print softmax.theta
    """
    """
    origin = bpnn(Train_xRot,train_Y[0],Test_xRot,test_Y[0])
    origin.Train(2000,10)
    pred = origin.Predict(Test_xRot)
    print 'nerual network error = ',float(sum(pred != test_Y[0])) / pred.shape[0]

    plt.plot(origin.errorrate[0], origin.errorrate[1],label = 'origin')
    plt.xlabel(r"Iter",fontsize=20)
    plt.ylabel(r'Error rate')
    plt.legend(loc = 'upper right')
    plt.title('Error rate vs iter ')
    plt.show()
    """