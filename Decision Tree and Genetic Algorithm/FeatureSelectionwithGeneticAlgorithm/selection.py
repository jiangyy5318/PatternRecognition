import numpy as np
def selection(pop,fvalue):
    fvalue = fvalue / np.sum(fvalue)
    fvalue = np.cumsum(fvalue, axis=0)
    ms = np.sort(np.random.rand(fvalue.shape[0],1),axis=0)
    fitin = 0
    newin = 0
    newpop = np.zeros(pop.shape,dtype=np.float64)
    while newin < pop.shape[0]:
        if(ms[newin,0] < fvalue[fitin,0]):
            newpop[newin,:] = pop[fitin,:]
            newin = newin + 1
        else:
            fitin = fitin + 1
    return newpop