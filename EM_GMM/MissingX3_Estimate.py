import numpy as np

if __name__ == "__main__":

    xyz_s = np.matrix([[0.42,-0.087,0.58],
            [-0.2,-3.3,-3.4],
            [1.3,-0.32,1.7],
            [0.39,0.71,0.23],
            [-1.6,-5.3,-0.15],
            [-0.029,0.89,-4.7],
            [-0.23,1.9,2.2],
            [0.27,-0.3,-0.87],
            [-1.9,0.76,-2.1],
            [0.87,-1,-2.6]])
    n = 10
    mu = np.matrix([[0.0],
                   [0.0],
                   [0.0]])


    # Normal
    mu_normal = np.transpose(xyz_s.sum(0))/n
    T = xyz_s - np.repeat(np.transpose(mu),10,axis = 0)
    Sigma_normal = np.dot(np.transpose(T),T)/n
    print "Normal"
    print mu_normal
    print Sigma_normal
    #print cccc
    sigma=np.matrix([[1.0,0,0],
           [0,1.0,0],
           [0,0,1.0]])
    for k in range(100):
        #Estep:
        sigma_inv = sigma#np.linalg.inv(sigma)
        sigma_z = 1/sigma_inv[2,2]
        for i in range(10):
            if(i%2 == 1):
                xyz_s[i,2] = mu[2,0] + sigma_z*sigma_inv[2,0:2]*(np.transpose(xyz_s[i,0:2])-mu[0:2,0])
        #Mstep
        T = xyz_s - np.repeat(np.transpose(mu),10,axis = 0)
        mu = np.transpose(xyz_s.sum(0))/n
        Sigma = np.dot(np.transpose(T),T)/n
        Sigma[2,2] += sigma_z/2
    print "EM"
    print mu
    print Sigma
