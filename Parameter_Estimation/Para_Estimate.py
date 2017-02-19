#-*- coding: UTF-8 -*-
from numpy import *
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
Accuracy = 0.001

def CDF_Random(CDF):
    Low = -10.0
    High = 10.0
    #print CDF
    Middle = (Low+High)/2
    MiddleCDF = CDF_Fucntion(Middle)
    while(abs(MiddleCDF - CDF)>Accuracy):
        if MiddleCDF > CDF:
            High = Middle
        else:
            Low = Middle
        Middle = (Low+High)/2
        MiddleCDF = CDF_Fucntion(Middle)
        #print Low,Middle,High,MiddleCDF
    return Middle
def CDF_Fucntion(x):
    #F(x) = 0.1F(（x+1）/1)+0.9F(（x-1）/1)
    return 0.1*norm.cdf(x+1)+0.9*norm.cdf(x-1)
    #return norm.cdf(x+1)

def PDF_Function(x):
    return 0.1*norm.pdf(x+1)+0.9*norm.pdf(x-1)
    #return norm.cdf(x+1)
def GenerateN_randomsDpCDF(N):
    randomArray = []
    while(N > 0):
        F = random.random()
        X = CDF_Random(F)
        randomArray.append(X)
        N-=1
        #print N
    return randomArray

def ParZenWindowFromInputArray(InputArray,Array,Para,ParZenFunction):
    Output = []
    for x in InputArray:
        if(ParZenFunction == WidthWidow):
            Output.append(ParZenWidthWindow(x,Array,Para))
        if(ParZenFunction == GaussWidow):
            Output.append(ParZenGaussWindow(x,Array,Para))
        if(ParZenFunction == Triangle):
            Output.append(ParZenTriangleKernel(x,Array,Para))
    return Output

def ParZenTriangleKernel(Input,Array,h):
    temp = 0.0
    ArrayLength = len(Array)
    for x in Array:
        if(abs(Input - x) <= h):
            temp += (h-abs(Input - x))/h/h
    return (temp/ArrayLength)

def ParZenWidthWindow(Input,Array,h):# h is width
    temp = 0.0
    ArrayLength = len(Array)
    for i in range(ArrayLength):
        if(abs(Input - Array[i]) <= h/2):
            temp += 1
    #print Input,temp
    return (temp/ArrayLength)/h

def ParZenGaussWindow(Input,Array,sigma):#Gauss window:input is x input,Array is sample, sigma is var
    temp = 0.0
    #print Input,sigma
    ArrayLength = len(Array)
    for i in range(ArrayLength):
        #print (Input - Array[i])*(Input - Array[i])
        #print 2*sigma^2
        temp += norm.pdf((Input - Array[i])*(Input - Array[i])/(2*sigma*sigma))
    return temp/ArrayLength

def FuncError(Estimate,Standard):# \sum_i^N(Estimate[i]-Standard[i])^2
    Length = len(Estimate)
    if(Length != len(Standard)):
        print "different length"
    #Difference = Standard - Estimate
    #print Difference
    a = matrix(Estimate)
    b = matrix(Standard)
    c = a-b
    SuareSum = sum(dot(c,transpose(c)))
    return SuareSum*0.02


def PlotToPic(Coordinate,InputX,InputY,StandardY,PicName):
    plt.figure(figsize=(4, 5))
    plt.plot(InputX, InputY, 'r.-')
    plt.plot(InputX, StandardY, 'g.-')
    plt.axis(Coordinate)
    #plt.xlabel("N = "+str(N[i])+", a = "+str(A[j]))
    plt.savefig(PicName+".png")
    #plt.show()
    plt.close()


def ParZenWindow(Cnt,ParZenFunction):
    # Cnt is a parameter,for different n and a, We will compute epsilon for Cnt time to obtain expectation and variance.
    # ParZenFunction = 1, width window
    # ParZenFunction = 2, Gauss window
    # ParZenFunction = 3, Triangle window
    N = [5,10,50,100,500,1000,5000,10000]
    A = [0.1,0.3,1.0,3.0]
    NLength = len(N)
    ALength = len(A)
    XArray = np.arange(-9.9,10, 0.2)
    YArray = PDF_Function(XArray)
    Error = matrix(zeros([Cnt,NLength*ALength]))
    for k in range(Cnt):
        for i in range(NLength):
            Samples = GenerateN_randomsDpCDF(N[i])
            for j in range(ALength):
                #Y2Array = ParZenWidthWindowFromInputArray(XArray,Samples,A[j])
                Y2Array = ParZenWindowFromInputArray(XArray,Samples,A[j],ParZenFunction)
                Error[k,(i*ALength)+j]  = FuncError(YArray,Y2Array)
                if(k == 0):#only when k == 0, we will plot estimated function and standard function
                                                                #picture name
                    PlotToPic([-5,5,0,0.4],XArray,YArray,Y2Array,"ParZen"+str(ParZenFunction)+"_"+str(i+1)+"_"+str(j+1))

    #print "Epsilon Average"
    epsilonAverage = np.sum(Error, axis=0)/Cnt

    #print epsilonAverage
    #print "Epsilon square Average"
    epsilonSquare = np.square(Error)
    epsilonSquareAverage = np.sum(epsilonSquare, axis=0)/Cnt
    #print epsilonSquareAverage
    #print "Diff"
    Var = epsilonSquareAverage - np.square(epsilonAverage)
    #print Var
    ResizeAverage = epsilonAverage.reshape((NLength,ALength))
    ResizeVar = Var.reshape((NLength,ALength))

    return ResizeAverage,ResizeVar


if __name__ == "__main__":

    #Calc()
    Cnull,WidthWidow,GaussWidow,Triangle = range(4)
    #print WidthWidow,GaussWidow,Triangle

    #Start Time
    starttime = datetime.now()
    print ParZenWindow(5,WidthWidow)
    endtime = datetime.now()
    print "WidthWidow Time: " , (endtime - starttime).seconds, 'seconds'


    starttime = datetime.now()
    print ParZenWindow(5,GaussWidow)
    endtime = datetime.now()
    print "GaussWidow Time: " , (endtime - starttime).seconds, 'seconds'


    starttime = datetime.now()
    print ParZenWindow(5,Triangle)
    endtime = datetime.now()
    print "Triangle Time: " , (endtime - starttime).seconds, 'seconds'


    #print "End"