from scipy.spatial import distance
import numpy as np
import scipy.io as sio
import datetime
import time
class knn:
    def __init__(self,data,label):
        self.data = data
        self.label = label
        self.Classes = len(np.unique(self.label))


    def KnnPredict(self,testdata,k,metric):

        (n,m) = testdata.shape
        pred = np.zeros((n,),dtype=int)
        Distance = distance.cdist(testdata,self.data,metric = metric)
        Select = Distance.argsort(axis=1,kind='quicksort')
        SelectLabel = self.label[Select[:,0:k]]

        for i in range(SelectLabel.shape[0]):
            A,B = np.histogram(SelectLabel[i],bins=range(10))
            pred[i] = np.argmax(A)
        return pred

    def ErrorRate(self,predict,testlabel):
        return float(sum(predict!=testlabel))/predict.shape[0]


if __name__ == "__main__":
    mat = sio.loadmat('pool5.mat')
    train_X = mat['train_X']
    train_Y = mat['train_Y']
    test_X = mat['test_X']
    test_Y = mat['test_Y']

    start = time.time()
    ccc = knn(train_X,train_Y[0])
    pred = ccc.KnnPredict(test_X,1,'euclidean')
    print 'pool5 Error rate :',ccc.ErrorRate(pred,test_Y[0])
    print 'pool5 Elapsed time',time.time() - start

    mat = sio.loadmat('fc6.mat')
    train_X = mat['train_X']
    train_Y = mat['train_Y']
    test_X = mat['test_X']
    test_Y = mat['test_Y']

    start = time.time()
    #print 'Elapsed Train time',TrainTime
    ccc = knn(train_X,train_Y[0])
    pred = ccc.KnnPredict(test_X,1,'euclidean')
    print 'fc6 Error rate :',ccc.ErrorRate(pred,test_Y[0])
    print 'fc6 Elapsed time',time.time() - start


