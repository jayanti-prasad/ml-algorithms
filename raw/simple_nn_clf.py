import argparse 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed (seed=1736)

"""
This demonstrate how to create a neural network with no hiddle layer 
and one output layer from scratch. 

This network can be used for regression purpose. 

"""

def sigmoid (y):
   return 1/ (1.0 + np.exp(-y)) 


def cross_entropy(y_true, y_predict):
    S = 0.0
    for i in range (0, len (y_true)):
       S = np.dot( y_true[i], np.log (y_predict[i])) #+ \
       #     np.dot( 1.0 - y_true[i], np.log (1.0 - y_predict[i]))
    return S




def grad(w, b, x):
    y = x.dot (w) + b 
    sigma = sigmoid (y) 

    grad_w = - x * sigma * (1.0 - sigma)
    grad_b = - sigma * (1.0 - sigma)
    return grad_w, grad_b 


def loss (y, y1):
    return  ((y-y1) ** 2) 


def grad_loss (w, b, x, y):
    y1  =  sigmoid (x.dot (w) +  b)
    grad_w, grad_b = grad(w, b, x)

    grad_w = -  (y-y1) * grad_w
    grad_b = -  (y-y1) * grad_b

    return grad_w, grad_b


def train_test_split(train_test_ratio, x, y):
    n = x.shape[0]

    r = np.random.random ([n])
    i_test  = [ i for i in range(0, n) if  r[i] < train_test_ratio]
    i_train = [ i for i in range(0, n) if i not in i_test]

    x_test = np.array([x[i,:] for i in i_test])
    y_test = np.array([y[i] for i in i_test])
 
    x_train = np.array([x[i,:] for i in i_train])
    y_train = np.array([y[i]   for i in i_train])
 
    return x_train, y_train, x_test, y_test 

    
class Dense:
   def __init__(self, dim_in, dim_out):
      self.dim_in = dim_in 
      self.dim_out = dim_out 
      self.W = None 
      self.b = None 

 
   def set_weights(self):

      func = np.random.random 
      #func = np.zeros 
          
      self.W = func([self.dim_in, self.dim_out])
      self.b = func([self.dim_out])

      print("Weights:", self.W.shape)
      print("Bias:", self.b.shape)
     

   def update(self, x):
      return  sigmoid (x.dot (self.W) +  self.b)



class Network:
    def __init__(self, dim_in, dim_out):
       self.dim_in = dim_in 
       self.dim_out = dim_out     

       self.L = Dense (self.dim_in,self.dim_out)
       self.L.set_weights ()


    def fit(self, x, y, niter, learning_rate):

       x_train, y_train, x_test, y_test = train_test_split(0.25, X, y)

       
       tr_loss = []
       ts_loss = []  

       for i in range (0, args.niter):

          # This loop is for training data 
          train_loss  = 0.0 
          for j in range(0, x_train.shape[0]):
              xx = x_train[j,:].reshape(x_train.shape[1],1)

              grad_w, grad_b = grad_loss (self.L.W, self.L.b, x_train[j,:], y[j])
              grad_w = grad_w.reshape([grad_w.shape[0],1]) 

              self.L.W = self.L.W + args.learning_rate * grad_w
              self.L.b = self.L.b + args.learning_rate * grad_b

              y1 = sigmoid (x_train[j,:].dot (self.L.W) +  self.L.b)

              train_loss = train_loss + loss (y_train[j], y1)
          
          # This loop is for testing data  
          test_loss = 0.0    
          for j in range (0, x_test.shape[0]):
              y1 = sigmoid (x_test[j,:].dot (self.L.W) +  self.L.b)
              test_loss = test_loss + loss (y_test[j], y1)
           
          train_loss = train_loss / y_train.shape[0]
          test_loss = test_loss   / y_test.shape[0]

          tr_loss.append (train_loss)
          ts_loss.append (test_loss)

       plt.plot(tr_loss, label="Train")
       plt.plot(ts_loss, label="Test")
       plt.legend()
       plt.show() 


    def predict (self, x):
        return  sigmoid (x.dot (self.L.W) +  self.L.b)
           

def input_func (X):
    return np.sqrt (np.mean(X))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--dim-input',type=int,help='Input dimensions',default=5)
    parser.add_argument('-o','--dim-output',type=int,help='output dimensions',default=1)
    parser.add_argument('-n','--niter',type=int,help='Number of iterations',default=100)
    parser.add_argument('-l','--learning-rate',type=float,help='Learning rate',default=0.01)

    args = parser.parse_args()

    # create data  
    X = np.random.random([100, args.dim_input])
    y = [input_func (X[i,:]) for i in range(0, X.shape[0])]


    # Get the network 
    N = Network (args.dim_input, args.dim_output)

    # Train the network 
    N.fit(X, y, args.niter, args.learning_rate)

    # Get some test data  
    x_test = np.random.random([20, args.dim_input])

    # Make prediction for test data  
    for i in range (0, x_test.shape[0]):
       print(x_test[i,:], N.predict (x_test[i,:])) 



