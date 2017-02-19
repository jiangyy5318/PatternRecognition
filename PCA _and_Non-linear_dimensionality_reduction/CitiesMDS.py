import numpy as np
from mds import mds
import matplotlib.pyplot as plt
import matplotlib.cm as cm
if __name__ == "__main__":
    Cities = np.array(['BeiJing','ShangHai','HeZe','GuangZhou','DaTong','XiAn','HerBin','NanJing'])
    D = np.matrix('0 2.16 4.68 3.33 5.85 2.08 1.91 2.08;0 0 12.95 2.41 9.33 2.58 2.91 1.08;0 0 0 17.91 11.76 9.56 21 8;0 0 0 0 36 2.58 4.16 2.25;0 0 0 0 0 16.26 24 16.88;0 0	0 0 0 0 2.91 2;0 0 0 0 0 0 0 2.83 ;0 0 0 0 0 0 0 0')
    X = mds((D+np.transpose(D))/2)
    ArrayX = np.array(X)
    colors = cm.rainbow(np.linspace(0,1,len(ArrayX)))
    for x,c,name in zip(ArrayX,colors,Cities):
        plt.scatter(x[0],x[1],label=name,color=c)
    plt.legend()
    plt.title('City locations under MDS')
    plt.show()
