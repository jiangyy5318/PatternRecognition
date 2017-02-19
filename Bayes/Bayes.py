import numpy as np

dir='data/'

#docIdx wordIdx count
def load(fType):
    trainData   = dir + fType + ".data"
    trainLabel  = dir + fType + ".label"
    train = fromfile(trainData, dtype=int, sep='\n')
    label = fromfile(trainLabel, dtype=int, sep='\n')
    train.shape = (-1,3)
    label.shape = (-1,1)

    return train,label

def ReadData(datadir,labeldir):


    return data,label