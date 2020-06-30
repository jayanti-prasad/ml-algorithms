import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import argparse 

def sigmoid (x):
    return 1/ (1+np.exp(-x))


def tanh(x):
    return np.tanh(x)


class LSTM :
    def __init__(self, dim_in, dim_out):

       self.dim_in = dim_in 
       self.dim_out = dim_out 
       np.random.seed (seed=192)

       [self.U_i, self.U_o, self.U_f, self.U_c] = [None] * 4
       [self.W_i, self.W_o, self.W_f, self.W_c] = [None] * 4
       [self.b_i, self.b_o, self.b_f, self.b_c] = [None] * 4

       self.c = None
       self.h = None

 
    def set_weights (self, seed):
      
       if seed :
          func = np.random.random
       else:
          func = np.zeros 

       # forget gate 
       self.W_f = func ([self.dim_out, self.dim_out])
       self.U_f = func ([self.dim_in, self.dim_out])
       self.b_f = func ([self.dim_out])

       # input gate 
       self.W_i = func ([self.dim_out, self.dim_out])
       self.U_i = func ([self.dim_in, self.dim_out])
       self.b_i = func ([self.dim_out])


       # memory gate 
       self.W_c = func ([self.dim_out, self.dim_out])
       self.U_c = func ([self.dim_in, self.dim_out])
       self.b_c = func ([self.dim_out])


       # output gate 
       self.W_o = func ([self.dim_out, self.dim_out])
       self.U_o = func ([self.dim_in, self.dim_out])
       self.b_o = func ([self.dim_out])

       # memory  
       self.c =  np.random.random([self.dim_out])
       self.h =  np.random.random([self.dim_out])


    def update (self, x):

       yi = sigmoid (x.dot (self.U_i) + self.h.dot (self.W_i) + self.b_i)
       yo = sigmoid (x.dot (self.U_o) + self.h.dot (self.W_o) + self.b_o)
       yf = sigmoid (x.dot (self.U_f) + self.h.dot (self.W_f) + self.b_f)
       yc =    tanh (x.dot (self.U_c) + self.h.dot (self.W_c) + self.b_c)

       self.c = self.c * yf + yi * yc 
       self.h = yo * tanh (self.c)

       return sigmoid (self.h)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--dim-input',type=int,help='Input dimensions')
    parser.add_argument('-o','--dim-output',type=int,help='output dimensions')

    args = parser.parse_args()

    L = LSTM (args.dim_input,args.dim_output)
    L.set_weights (129)

    print("num params:", 4 * (L.U_i.shape[0]* L.U_i.shape[1] + L.W_i.shape[0]* L.W_i.shape[1] +L.b_i.shape[0]))

    X = np.random.random([20, args.dim_input])

    H = []
    for i in range(0, X.shape[0]):
       output =  L.update (X[i,:])
       H.append (output)

    H = np.array (H)
    print(H)
    
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot()
      
    ax.imshow(H.T, cmap='hot', interpolation='nearest')
    plt.show()

