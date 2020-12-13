import numpy as np


def conv (X, f):

    (nx, ny) = X.shape[0], X.shape[1]
    print(nx, ny)
 
    y = np.zeros([X.shape[0]-fx, X.shape[1]-fx. X.shape[2]])

    for i in range (fx, nx-fx):
       for i in range (fx, nx-fx):
           y[i,j,:] = X [i, j, :] * f[i,



if __name__ == "__main__":

    X = np.random.random([28,28,1])

    f = np.zeros([2,2])

    conv(X, f)
