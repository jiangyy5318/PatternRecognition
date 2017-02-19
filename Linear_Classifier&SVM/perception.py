#-*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from numpy.matlib import repmat

def randomdata(N,L):
    #data = np.zeros((N,3))
    #L is a list
    #L[0]*x + L[1]*y = L[2] and L[0]*x + L[1]*y = L[2]
    data = np.random.random_sample((N, 2))*20-10
    label = L[0]*data[:,0]+L[1]*data[:,1]
    for i in range(label.shape[0]):
        if label[i] >= L[2] :label[i] = 1
        elif label[i] <= L[3]:label[i] = -1
        else:label[i] = 0
    return data[np.array(np.where(label!=0)),:][0],label[np.array(np.where(label!=0))].transpose()

def Perceptron(data,label,margin = 1):
    data = np.concatenate((data, np.ones((data.shape[0],1))), axis=1)
    alpha = np.matrix([[0.0],[0.0],[0.0]])
    k = 0
    nLength = data.shape[0]

    distance = np.dot(data, alpha) * label
    while(1):
        k = (k+1)%nLength
        if(distance[k,0] <= margin):
            alpha = alpha + np.transpose(data[k,:])
            distance = np.dot(data, alpha) * label
        if(np.min(distance) > margin):
            break
    return alpha


if __name__ == "__main__":
    data,label = randomdata(50,[1,1,1,-1])
    print label.shape
    for x,c in zip(data,label):
        plt.scatter(x[0],x[1],c='b' if c == 1 else 'r')

    alpha = Perceptron(data,label,margin=0)
    alpha2 = Perceptron(data,label,margin=1)
    X = np.arange(-10, 10, 1)
    Y = np.arange(-10, 10, 1)
    x, y = np.meshgrid(X, Y)
    f = np.array(alpha)[0][0] * x+ np.array(alpha)[1][0] * y + np.array(alpha)[2][0]
    f2 = np.array(alpha2)[0][0] * x+ np.array(alpha2)[1][0] * y + np.array(alpha2)[2][0]
    plt.contour(x, y, f, 0, colors = 'r')
    plt.contour(x, y, f2, 0, colors = 'b')
    red_line = mlines.Line2D([], [], color='red', marker='*',
                          markersize=15, label='Fixed')
    blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='With margin')
    plt.legend(handles=[red_line,blue_line])
    plt.show()
