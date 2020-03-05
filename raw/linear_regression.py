import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
This program fits a line to a simulated data set.
There are only two hyper-parameters : the number of
iterations and the learning rate. 

  - Jayanti Prasad (prasad.jayanti@gmail.com)
    Feb 05, 2020
"""


def line (w, b, x):
    return np.dot (x, w) + b


def sim_data (num_data, num_dim):
   
   #Creating simulate data for the linear fit.
   #One can make the problem challenging by adding more noise.
   
   # Let us set the intercept & slope  
   # This is a one dimensional fit 
   
   w = np.random.randint(10,size=(num_dim))
   b = np.random.randint(10,size=(1))
   x = np.random.random([num_data, num_dim])
   e = np.random.normal(0.0,0.5,num_data)
   y = line(w, b, x[:,]) + e

   return x, y


class LinearFit:
   # This is the main fitting module 

   def __init__(self, args, ndim):
      # set the weight and intercept randomly 
      self.w = np.random.random (ndim)
      self.b = np.random.random (1)

      self.lr = float (args.learning_rate)  
      self.iter = int (args.num_iter)


   def __grad__ (self, x,y):
      # This is the gradiant of the loss function MSE 
      e = y - line (self.w, self.b, x[:,])
      grad_w = np.dot(x.T, e) /y.shape[0]
      grad_b = np.sum(e) / y.shape[0]
      return grad_w, grad_b

   def __loss__ (self, x, y):
      e = y - line (self.w, self.b, x[:,])
      return np.dot (e, e) / e.shape[0]


   def fit(self, x, y):
      # This is the main fitting module
      for i  in range(0, self.iter):
         grad_w, grad_b  = self.__grad__(x, y)
         self.w = self.w + self.lr * grad_w
         self.b = self.b + self.lr * grad_b
         print("iter:", i, "loss:", self.__loss__(x, y))
        
   def predict (self, x):
       return  line (self.w, self.b, x)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iter', help='Number of iterations', required=True)
    parser.add_argument('-lr', '--learning-rate', help='Learning Rate', required=True)
    parser.add_argument('-d', '--num-dim', help='Dimensions', required=True)
    args = parser.parse_args()

    d = int (args.num_dim) 
    x, y = sim_data(1000,d) 

    L = LinearFit(args, d)

    L.fit(x,y)

    y_hat = L.predict(x[:,])
 
    plt.plot(x[:,0],y,'o')
    plt.plot(x[:,0],y_hat,'+')
    plt.ylim(np.min(y),np.max(y))
    plt.show()  
