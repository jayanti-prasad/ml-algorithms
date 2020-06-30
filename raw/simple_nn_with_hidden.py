import numpy as np
import argparse
import matplotlib.pyplot as plt 
np.random.seed (seed =282)

"""

- This program builds a neural network from scratch (using numpy only)
without using any higher level library. 


- This is program to show how feed-forward neural network with one 
hidden layer can be used to discover a non-linear mapping between the
input and output data.

- This program is just for demonstration and may work only with the
data used here. For any different case, changes may be needed particularly
in the hyper-parameters.

- In case you find any bug in the program please let me know.

-  Jayanti Prasad [prasad.jayanti@gmail.com]

"""


def sigmoid (x):
   """
   Activation function, used for all the layers 
   """
   return 1.0/ (1.0 + np.exp(-x))


def dense (W, b, X):
   """
   Non-linear mapping 
   """
   y =  X.dot (W) + b
   return sigmoid (y)


def loss (y, y1):
    """ 
    Loss function, for actual see below 
    """
    return (y-y1)**2 


def train_test_split(train_test_ratio, x, y):

    """
    For splitting data into traing and testing 
    """

    n = x.shape[0]
    r = np.random.random([n]) 
    i_test  = [ i for i in range(0, n) if  r[i] > train_test_ratio]
    i_train = [ i for i in range(0, n) if i not in i_test]

    x_test = np.array([x[i,:] for i in i_test])
    y_test = np.array([y[i] for i in i_test])

    x_train = np.array([x[i,:] for i in i_train])
    y_train = np.array([y[i]   for i in i_train])

    return x_train, y_train, x_test, y_test



class Network:
   """
   This is the main network which has one input, one hidden and 
   one output layer. Output layer has just one neuron so this
   network can be used for regression.


   """
   def __init__(self, dim_in, dim_hid, dim_out):
       self.dim_in = dim_in 
       self.dim_hid = dim_hid
       self.dim_out = dim_out 

       self.W1 = np.random.random([self.dim_in, self.dim_hid])
       self.b1 = np.random.random([self.dim_hid])

       self.W2 = np.random.random([self.dim_hid, self.dim_out])
       self.b2 = np.random.random([self.dim_out])
           

   def forward (self, X):
       """ 
       This is forward loop.
       """

       y1 = dense (self.W1, self.b1, X)
       output = dense (self.W2, self.b2, y1)
       return y1, output 


   def backprop(self, X, y, y1, output):

       """
       This is back propagation loop.
       """
       error  = y - output
 
       sigma  = sigmoid (output)
       sigma1  = np.array([ sigmoid (y1[i]) for i in range (0, y1.shape[0])])
    
       grad_w2 = - 2 * error * sigma * (1-sigma) * y1 
       grad_b2 = - 2 * error * sigma * (1-sigma) 
   
       grad_w1 = -2 * error  * sigma * (1-sigma) * np.outer (X, sigma1 * (1-sigma1)) 
       grad_b1 = -2 * error  * sigma * (1-sigma) * sigma1 * (1-sigma1)  

       grad_w2 = grad_w2.reshape(grad_w2.shape[0],1)      

       return grad_w2, grad_b2, grad_w1, grad_b1 



   def train(self, X, y, num_iter, alpha, train_test_ratio):
       """
       Train the network  
       """
       x_train, y_train, x_test, y_test = train_test_split(train_test_ratio, X, y)

       tr_loss  = [] 
       ts_loss  = [] 

       for i in range (0, num_iter):

           # training loop 
           train_loss = 0 
           for j in range (0, x_train.shape[0]):
               y1, output = self.forward (x_train[j,:])
               grad_w2, grad_b2, grad_w1, grad_b1 = self.backprop(x_train[j,:], y_train[j], y1, output)
               self.W2 = self.W2 - alpha * grad_w2  
               self.W1 = self.W1 - alpha * grad_w1  
               self.b1 = self.b1 - alpha * grad_b1  
               self.b2 = self.b2 - alpha * grad_b2  
               train_loss += loss(output[0], y_train[j]) / x_train.shape[0] 
           tr_loss.append (train_loss)

           # testing loop 
           test_loss = 0  
           for j in range (0, x_test.shape[0]):
               y1, output = self.forward (x_test[j,:])
               test_loss += loss(output[0], y_test[j]) / x_test.shape[0] 

           ts_loss.append (test_loss)

           print(i, train_loss, test_loss)

       plt.plot(tr_loss,label='Train')
       plt.plot(ts_loss,label='Test')
       plt.legend()

       plt.show()


 
def input_func (X):
    """
    For creating data 
    """

    #return  (np.mean(X))
    return np.sqrt (np.mean(X))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--dim-in',type=int,help='Input dimensions',default=10)
    parser.add_argument('-m','--dim-hid',type=int,help='Hidden dimensions',default=6)
    parser.add_argument('-o','--dim-out',type=int,help='output dimensions',default=1)
    parser.add_argument('-n','--num-iter',type=int,help='Number of iterations',default=100)
    parser.add_argument('-l','--learning-rate',type=float,help='Learning rate',default=0.01)
    parser.add_argument('-t','--train-test-split',type=float,help='Train test ratio',default=0.75)

    args = parser.parse_args()

    X = np.random.random ([100, args.dim_in])
    y = np.array ([input_func (X[j,:]) for j in range (0, X.shape[0])])

    N = Network (args.dim_in, args.dim_hid, args.dim_out)

    N.train (X, y, args.num_iter, args.learning_rate, args.train_test_split)
  
    # Make prediction using trained network 

    xx = np.random.random ([100, args.dim_in])
    yy = np.array ([input_func (xx[j,:]) for j in range (0, xx.shape[0])])

    print("Prediction")      
    for i in range (0, yy.shape[0]):
       y1, output = N.forward (xx[i,:])         
       print(yy[i], output)

