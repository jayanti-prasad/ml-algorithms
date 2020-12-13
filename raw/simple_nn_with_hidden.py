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


def loss (y_true, y_pred):
    return np.sum  ( (y_true -y_pred) ** 2) / y_true.shape[0]


def test_train_split(test_frac, x, y):
    """
    For splitting data into traing and testing 
    """
    n = x.shape[0]
    r = np.random.random([n]) 
    i_test  = [ i for i in range(0, n) if  r[i] < test_frac]
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
   def __init__(self, dim_in, dim_out):
       self.dim_in = dim_in 
       self.dim_hid = 10
       self.dim_out = dim_out 

       self.W1 = np.random.random([self.dim_in, self.dim_hid])
       self.b1 = np.random.random([self.dim_hid])

       self.W2 = np.random.random([self.dim_hid, self.dim_out])
       self.b2 = np.random.random([self.dim_out])
           

   def forward (self, X):
       """ 
       This is forward loop.
       """
       y_hid = sigmoid ( X.dot (self.W1) + self.b1)
       y_out = sigmoid ( y_hid.dot (self.W2) + self.b2)
       return y_hid, y_out  


   def backprop(self, X, y, y_hid, y_pred, alpha):

       """
       This is back propagation loop.
       """
       error  = y - y_pred
 
       sigma_out  = sigmoid (y_pred)
       sigma_hid  = np.array([ sigmoid (y_hid[i]) \
          for i in range (0, y_hid.shape[0])])
    
       grad_w2 = - 2 * error * sigma_out * (1-sigma_out) * y_hid
       grad_b2 = - 2 * error * sigma_out * (1-sigma_out) 
   
       grad_w1 = -2 * error  * sigma_out * (1-sigma_out) * \
           np.outer (X, sigma_hid * (1-sigma_hid)) 

       grad_b1 = -2 * error  * sigma_out * (1-sigma_out) * \
           sigma_hid * (1-sigma_hid)  

       grad_w2 = grad_w2.reshape(grad_w2.shape[0],1)      

       self.W2 = self.W2 - alpha * grad_w2
       self.W1 = self.W1 - alpha * grad_w1
       self.b1 = self.b1 - alpha * grad_b1
       self.b2 = self.b2 - alpha * grad_b2
       

   def train(self, X, y, num_iter, alpha, val_frac):
       """
       Train the network  
       """
       self.alpha = alpha 

       X_train, y_train, X_val, y_val = test_train_split(val_frac, X, y)

       tr_loss  = [] 
       vl_loss  = [] 

       for i in range (0, num_iter):

           # training loop 
           y_train_pred = np.zeros ([X_train.shape[0]])

           for j in range (0, X_train.shape[0]):
               # forward pass 
               y_hid,  y_train_pred [j] = self.forward (X_train[j,:])

               # backward pass  
               self.backprop(X_train[j,:], y_train[j],\
                  y_hid, y_train_pred[j], alpha)

           train_loss = loss(y_train_pred, y_train) 
           tr_loss.append (train_loss)
 
           # testing loop
           
           y_val_pred = np.zeros ([X_val.shape[0]])

           for j in range (0, X_val.shape[0]):
               y_hid, y_val_pred[j] = self.forward (X_val[j,:])

           val_loss = loss (y_val_pred, y_val)
           vl_loss.append (val_loss)
           print(i, train_loss, val_loss)

       return tr_loss, vl_loss  


def get_data (n, dim_input):
    # create data  

    # This is the maping between input vector and output scalar 

    input_func = lambda x : np.sqrt (np.mean(x))

    X = np.random.random([n, dim_input])
    y = [input_func (X[i,:]) for i in range(0, X.shape[0])]

    return X, y
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--dim-in',type=int,help='Input dimensions',default=10)
    parser.add_argument('-o','--dim-out',type=int,help='output dimensions',default=1)
    parser.add_argument('-n','--num-iter',type=int,help='Number of iterations',default=100)
    parser.add_argument('-l','--learning-rate',type=float,help='Learning rate',default=0.01)
    parser.add_argument('-t','--test-frac',type=float,help='Train test ratio',default=0.25)
    parser.add_argument('-v','--val-frac',type=float,help='Train Validation ratio',default=0.10)

    args = parser.parse_args()

    X, y = get_data (100, args.dim_in)

    X_train, y_train, X_test, y_test = test_train_split(args.test_frac, X, y)

    N = Network (args.dim_in, args.dim_out)

    # Train the network  
    tr_loss, vl_loss = N.train (X_train, y_train, args.num_iter,\
       args.learning_rate, args.val_frac)


    # Make prediction

    y_test_pred = np.zeros ([X_train.shape[0]])

    for i in range (0, X_train.shape[0]):
        y_hid, y_test_pred[i] = N.forward (X_train[i,:])

    # Plot prediction & Loss 
 
    fig = plt.figure (figsize=(12,9))
    ax = fig.add_subplot(121)      
    bx = fig.add_subplot(122)      

    ax.plot(y_train, label='True')
    ax.plot(y_test_pred, label='Predicted')
    ax.legend()   

    bx.plot(tr_loss,label='Training data')
    bx.plot(vl_loss,label='Validation  data')
    bx.legend()

    plt.show()
