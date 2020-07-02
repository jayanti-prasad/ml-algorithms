import argparse 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed (seed=1736)

"""
This demonstrate how to create a neural network with no hiddle layer 
and one output layer from scratch. 

This network can be used for regression purpose. 

"""


def sigma (x):
   return 1/ (1.0 + np.exp(-x)) 


def grad_sigma(w, b, x):

    s = sigma ( x.dot (w) + b) 
    grad_w = - s * (1.0 - s) * x 
    grad_b = - s * (1.0 - s)
    return grad_w, grad_b 


def loss (y_true, y_pred):
     return np.sum  ( (y_true -y_pred) ** 2) / y_true.shape[0]


def grad_loss (w, b, x, y, y_pred):

    grad_w, grad_b = grad_sigma (w, b, x)

    grad_w = -  (y-y_pred) * grad_w
    grad_b = -  (y-y_pred) * grad_b

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
      self.W = np.random.random([dim_in, dim_out])
      self.b = np.random.random([dim_out])

   def out(self, x):
      return  sigma(x.dot (self.W) +  self.b)



class Network:
 
    def __init__(self, dim_in, dim_out):
       self.L = Dense (dim_in,dim_out)


    def fit(self, X, y, niter, learning_rate, val_split):

       x_train, y_train, x_val, y_val = train_test_split(val_split, X, y)

       tr_loss = []
       vl_loss = []  
       for i in range (0, args.niter):

          # This loop is for training data 
          y_train_pred = np.zeros (y_train.shape[0])
          y_val_pred   = np.zeros (y_val.shape[0])
 
          for j in range(0, x_train.shape[0]):

              # predict y 
              y_train_pred[j] = self.L.out(x_train[j,:])

              # get the gradients  
              grad_w, grad_b = grad_loss (self.L.W, self.L.b, x_train[j,:], y[j], y_train_pred[j])
              grad_w = grad_w.reshape([grad_w.shape[0],1]) 

              # update the weights 
              self.L.W = self.L.W + args.learning_rate * grad_w
              self.L.b = self.L.b + args.learning_rate * grad_b
        
          # get the loss for training data
          train_loss = loss (y_train, y_train_pred)
          tr_loss.append (train_loss)
          
  
          # This loop is for validation  data  
          for j in range (0, x_val.shape[0]):
              y_val_pred[j]  = self.L.out(x_val[j,:])
          val_loss = loss (y_val, y_val_pred)
          vl_loss.append (val_loss)

       plt.plot(tr_loss, label="Train data")
       plt.plot(vl_loss, label="Validation data")
       plt.ylabel("Loss")
       plt.xlabel("Iterations")
       plt.legend()
       plt.show() 


    def predict (self, x):
        return  self.L.out(x)

# create fake data 

           
def input_func (X):
    return np.sqrt (np.mean(X))

def get_data (n):
    # create data  
    X = np.random.random([n, args.dim_input])
    y = [input_func (X[i,:]) for i in range(0, X.shape[0])]

    return X, y 



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--dim-input',type=int,help='Input dimensions',default=5)
    parser.add_argument('-o','--dim-output',type=int,help='output dimensions',default=1)
    parser.add_argument('-n','--niter',type=int,help='Number of iterations',default=100)
    parser.add_argument('-l','--learning-rate',type=float,help='Learning rate',default=0.01)
    parser.add_argument('-t','--test-split',type=float,help='Test split',default=0.20)
    parser.add_argument('-v','--validation-split',type=float,help='Validation split',default=0.10)

    args = parser.parse_args()

    X, y = get_data (100)

    # split the data into test & train 

    X_train, y_train, X_test, y_test = train_test_split(args.test_split,  X, y)

    # Get the network 
    N = Network (args.dim_input, args.dim_output)

    # Train the network 
    N.fit(X_train, y_train,  args.niter, args.learning_rate, args.validation_split)

    y_test_predict = [N.predict (X_test[i,:]) for i in range (0, y_test.shape[0])]

    print("testing loss=", loss (y_test, y_test_predict))

    plt.plot(y_test, label='True [Testing data]')
    plt.plot(y_test_predict, label='Predicted [Testing data]')
    plt.legend()
    plt.show()
    

