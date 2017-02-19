import scipy.io as sio
import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def Choose01From0_9(X,Y):
    TransposeY = np.transpose(Y)
    NewX = []
    NewY = []
    for XElement,YElement in zip(X,TransposeY):
        if YElement[0]<= 1:
            NewX.append(XElement)
            #print XElement
            NewY.append(YElement)
    #print XElement
    return np.matrix(NewX),np.matrix(NewY)

def dotproduct(A,B):
    AAAA =  np.dot(np.matrix(A),np.transpose(B))
    #print AAAA.shape
    #print np.array(AAAA)[0][0]
    return np.array(AAAA)[0][0]

def pocket(E,C,max_iter,TestE,TestC):
    (num,dim) = E.shape
    (test_num,test_dim) = TestE.shape
    pi = np.zeros((1,dim))
    w = np.zeros((1,dim))
    run_pi = 0
    run_w = 0
    num_ok_pi = pocket_Check_num(E,C,pi)
    num_ok_w= pocket_Check_num(E,C,w)
    #error = np.zeros(max_iter,2)
    iter_num = []
    error_test=[]
    error_train=[]

    for iter in range(1,max_iter):
        k = random.randint(0,num-1)
        print iter,k
        #print pi
        #print E[iter,:]
        #print pi.shape
        #print E[iter,:].shape
        #print C.shape
        if (dotproduct(pi,E[iter,:]) > 0 and C[iter,0] == 1) or (dotproduct(pi,E[iter,:]) < 0 and C[iter,0] == 0):
            run_pi += 1
            if run_pi > run_w:
                num_ok_pi = pocket_Check_num(E,C,pi)
                if num_ok_pi >= num_ok_w:
                    w = pi
                    run_w = run_pi
                    num_ok_w = num_ok_pi
                    if num_ok_w == num:
                        break
        else:
            pi += (C[k][0] - 0.5)*2*E[k,:]
            print pi
            run_pi = 0

        num_ok = pocket_Check_num(TestE,TestC,w)
        iter_num.append(iter)
        print num_ok
        error_test.append((float)(test_num-num_ok)/test_num)
        error_train.append((float)(num-num_ok_w)/num)

    return w,iter_num,error_test,error_train
    #return
def pocket_Check_num(E,C,pi):

    (num,dim) = E.shape
    num_ok_pi = 0
    for i in xrange(num):
        if (dotproduct(pi,E[i,:]) > 0 and C[i,0] == 1) or (dotproduct(pi,E[i,:]) < 0 and C[i,0] == 0):
            num_ok_pi += 1
    return num_ok_pi


if __name__ == "__main__":
    print 'Start'
    data=sio.loadmat('mnist.mat')
    Train_X,Train_Y = Choose01From0_9(data['train_X'],data['train_Y'])
    Test_X,Test_Y=Choose01From0_9(data['test_X'],data['test_Y'])
    #print Train_X.shape
    Train_ConcatenateOnes = np.matrix(np.ones((Train_X.shape[0],1)))
    NewTrain_X = np.concatenate((Train_ConcatenateOnes,Train_X),axis=1)
    Train_ConcatenateOnes = np.matrix(np.ones((Test_X.shape[0],1)))
    NewTest_X = np.concatenate((Train_ConcatenateOnes,Test_X),axis=1)
    #print NewTrain_X.shape
    (W,iter_num,error_test,error_train) = pocket(NewTrain_X,Train_Y,100,NewTest_X,Test_Y)
    print error_test
    print error_train
    plt.plot(iter_num,error_test, 'r')
    plt.plot(iter_num,error_train,'b')
    red_patch = mpatches.Patch(color='red', label='Test')
    blue_patch = mpatches.Patch(color='blue', label='train')
    plt.legend(handles=[red_patch,blue_patch])
    plt.show()
    print 'End'