import numpy as np


np.random.seed (seed=234)


def sigma (x):
   return 1/(1+np.exp(-x))


def fibo(n):
   x = [0,1]

   for i in range (2, n):
      x.append ( x[i-1] + x[i-2])

   return x 


class RNN :
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U = None
        self.W = None
        self.b = None 
        
        self.h = None 

    def set_weights (self):

        self.U = np.random.random([self.dim_in, self.dim_out])
        self.W = np.random.random([self.dim_out, self.dim_out])

        self.b = np.random.random([self.dim_out])

        self.h = np.random.random([self.dim_out])

    def update (self, x): 
        self.h = sigma (x.dot (self.U) + self.h.dot (self.W) + self.b)

        return self.h 


if __name__ == "__main__":
   
    dim_in = 10
    dim_out = 6 

    d0 = np.random.randint(100,size=[1000])

    print(d0)


                 



