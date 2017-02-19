import numpy as np
from crossover import crossover
from mutation import mutation
from selection import selection
import matplotlib.pyplot as plt
#def objective(pop):
#    x np.dot(pop,b2int).astype(np.float64)*10/1024

def fitvalue(objvalue,threhold):
    return np.maximum(objvalue,np.ones(objvalue.shape,dtype=np.float64)*threhold)

def f(temp):
    return 10*np.sin(5*temp)+7*np.cos(4*temp)


popsize = 20
chromlength = 10

pop = np.random.randint(2, size=(popsize, chromlength))
#print pop
b2int = np.power(2, range(chromlength)).reshape(chromlength,-1)
#print b2int

#print np.dot(pop,b2int)
#print pop.shape
pc = 0.6 # cross-over probality
pm = 0.6 # mutation probab

maxIter = 20
x = np.zeros((maxIter))
y = np.zeros((maxIter))

for iter in range(maxIter):
    #objvalue = objective(pop)
    temp = np.dot(pop,b2int).astype(np.float64)*10/1024
    objvalue = f(temp)

    fvalue = fitvalue(objvalue,0.1)

    argmax = np.argmax(fvalue,axis=0)[0]
    print argmax
    x[iter] = temp[argmax,0]
    y[iter] = fvalue[argmax,0]

    newpop = selection(pop,fvalue)

    newpop = crossover(newpop,pc)

    newpop = mutation(newpop,pm)

    pop = newpop

Xline = np.arange(0.0, 10.0, 0.1)
Yline = f(Xline)
line, = plt.plot(Xline, Yline, lw=2)
plt.plot(x,y,'r*')
plt.show()