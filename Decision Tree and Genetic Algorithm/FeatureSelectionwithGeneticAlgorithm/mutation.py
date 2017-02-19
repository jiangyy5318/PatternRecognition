import numpy as np
import random
def mutation(pop,pm):
    #gerate new individuals by the way of mutation
    #Inputs: pop,prob
    #output newpop
    newpop = pop.copy()
    for i in range(newpop.shape[0]):
        if random.random() < pm:
            index = np.random.randint(pop.shape[1], size=2)
            newpop[i,index[0]] = pop[i,index[1]]
            newpop[i,index[1]] = pop[i,index[0]]
    return newpop