import numpy as np
from crossover import crossover
from mutation import mutation
from selection import selection
import scipy.io as sio
from sklearn import svm
import matplotlib.pyplot as plt
def objective(pop,train_X,train_Y):
    objvalue = np.zeros((pop.shape[0],1))
    for i in range(pop.shape[0]):
        f = np.where(pop[i,:]==1)
        clf = svm.SVC()
        X = train_X[:,f[0]]
        clf.fit(X, train_Y)
        objvalue = clf.score(X, train_Y, sample_weight=None)
    return objvalue


def fitvalue(objvalue,threhold):
    return np.maximum(objvalue,np.ones(objvalue.shape,dtype=np.float64)*threhold)

#N = 200
mat = sio.loadmat('mnist.mat')
chooseTrain = np.random.permutation(np.where(np.array(mat['train_Y'][0]) < 2))[0:1000]
chooseTest = np.random.permutation(np.where(np.array(mat['test_Y'][0]) < 2))[0:1000]
train_X = mat['train_X'][chooseTrain,:][0]
train_Y = mat['train_Y'][0,chooseTrain].transpose()
test_X = mat['test_X'][chooseTest,:][0]
test_Y = mat['test_Y'][0,chooseTest].transpose()



popsize = 20
chromlength = train_X.shape[1]

pop = np.random.randint(2, size=(popsize, chromlength))

#print pop.shape
pc = 0.6 # cross-over probality
pm = 0.6 # mutation probab

maxIter = 20
x = np.zeros((maxIter))
y = np.zeros((maxIter))
feature = np.zeros((maxIter,chromlength))


for iter in range(maxIter):
    #objvalue = objective(pop)
    #temp =
    objvalue = objective(pop,train_X,train_Y)

    fvalue = fitvalue(objvalue,0.3)

    argmax = np.argmax(fvalue,axis=0)[0]
    y[iter] = fvalue[argmax,0]
    feature[iter,:] = pop[argmax,:]

    newpop = selection(pop,fvalue)

    newpop = crossover(newpop,pc)

    newpop = mutation(newpop,pm)

    pop = newpop

print y
clf = svm.SVC()
f = np.which(feature[maxIter-1,:]==1)
X = test_X[:, np.array(f)[0]]
#print X.shape
clf.fit(X, test_Y)
print clf.score(X, y, sample_weight=None)

