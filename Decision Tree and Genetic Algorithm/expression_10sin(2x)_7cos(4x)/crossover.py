import numpy as np
import random
def crossover(pop,pc):
    #gerate new individuals by the way of crossover
    #Inputs: pop,pc
    #output newpop
    #print pop
    newpop = pop.copy()
    for i in range(0,newpop.shape[0],2):
        if random.random() < pc:
            index = np.random.randint(pop.shape[1], size=2)
            #print index
            #print pop[i,0:index[0]],pop[i+1,index[0]:]
            newpop[i,index[0]:] = pop[i+1,index[0]:]#np.concatenate((pop[i,0:index],pop[i+1,index:]),axis=0)
            newpop[i+1,index[0]:] = pop[i,index[0]:]#np.concatenate((pop[i+1,0:index],pop[i,index:]),axis=0)
    return newpop