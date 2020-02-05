import sys
import numpy as np
import matplotlib.pyplot as plt

"""
This program fits a line to a simulated data set.
There are only two hyper-parameters : the number of
iterations and the learning rate. 

  - Jayanti Prasad (prasad.jayanti@gmail.com)
    Feb 05, 2020
"""


def func (x, w, b):
   return w *x + b

def error(x, w, b, y):
    return y - func (x, w, b)

def grad (x, w, b, y):
    e = error (x, w, b, y)
    grad_w, grad_b = np.sum(e*x), np.sum(e) 
    return grad_w/len(x), grad_b/len(x) 


class LinearFit:

   def __init__(self, niter, lr ):

      r = np.random.random(2)
      self.w = r[0]
      self.b = r[1]
      self.lr = lr  
      self.niter = niter 


   def fit(self, x, y):

      for i  in range(0, niter):
         grad_w, grad_b  = grad(x, self.w, self.b, y)
         self.w = self.w + self.lr * grad_w
         self.b = self.b + self.lr * grad_b
         print(i, self.w, self.b, grad_b) 

        
def get_data ():
   b, w = 100.0, 3.0
   x = np.random.random(100)
   e = np.random.normal(0.0,0.5,100)
   y = w * x + b + e
   return x, y
  
   
if __name__ == "__main__":
   x,  y = get_data() 

   if len (sys.argv) > 1:
      niter, lr = int(sys.argv[1]), float(sys.argv[2])
   else: 
      niter, lr = 1000, 0.1 

   L = LinearFit(niter, lr )

   L.fit(x,y)

   y_hat = L.w *x + L.b 

 
   plt.plot(x,y,'o')
   plt.plot(x,y_hat)
   plt.show()  
