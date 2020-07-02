import numpy as np
import sys
import argparse 
import matplotlib.pyplot as plt 

np.random.seed (seed=1929)

def binary_cross_entropy (y_true, y_pred):
    if np.argmax (y_true) == 0:
       return - np.log (y_pred[1])
    else :
       return - np.log (y_pred[0])


def sigma (x):
    return 1.0 /(1+np.exp (-x))


def grad (w, b, x, y):
    y_true = np.argmax (y)
    y_p =  sigma (w * x + b)
    grad_w = - (y_true - y_p) * x
    grad_b = - (y_true - y_p) 
    return grad_w, grad_b 


def get_data (n, noise):
    X = np.random.random ([n])
    D = {'0' : [1,0], '1': [0,1]}
 
    s = np.random.random ([100])

    y = [[0,0] for i in range (0,100)]
    for i in range (0, X.shape[0]):
       if (X[i] > 0.5)  and s[i] < 1 - noise :
          y[i] = D['1']
       else: 
          y[i] = D['0']
    
    return X, y 


def plot_data (w, b, X, y, train_loss, train_acc):
     
    x1 = [X[i] for i in range (0, X.shape[0]) if np.argmax (y[i]) == 0]
    x2 = [X[i] for i in range (0, X.shape[0]) if np.argmax (y[i]) == 1]

    y1 = [0] * len (x1)
    y2 = [0] * len (x2)

    y1p = [ sigma (x1[i] * w + b)  for i in range (0, len (y1))]
    y2p = [ sigma (x2[i] * w + b)  for i in range (0, len (y2))]

    fig = plt.figure (figsize=(12,9))
    ax = fig.add_subplot(131)
    bx = fig.add_subplot(132)
    cx = fig.add_subplot(133)


    bx.set_ylabel("Loss [Binary Cross Entropy]")
    bx.set_xlabel("time")

    cx.set_ylabel("Accuracy")
    cx.set_xlabel("time")

    ax.set_ylabel("y=p(x)")
    ax.set_xlabel("x")

    ax.plot(x1, y1p,'ro')
    ax.plot(x2, y2p,'bo')

    ax.plot(x1, y1,'rx')
    ax.plot(x2, y2,'bx')

    bx.plot(train_loss)

    cx.plot(train_acc)

    plt.show()


def metrics (y_true, ss):

   tp, tn, fp, fn  = 0, 0, 0, 0

   if np.argmax (np.array(y_true)) == 0:
      if ss <= 0.5:
          tn = 1
      else:
          fn = 1
   
   if np.argmax (np.array(y_true)) == 1: 
      if ss >= 0.5:
          tp = 1
      else:
          fp = 1 

   return tp, fp, tn, fn 


if __name__ == "__main__":
    
     parser = argparse.ArgumentParser()
     parser.add_argument('-n','--num-iter',type=int,help='Number of iterations', default=100)
     parser.add_argument('-a','--learning-rate',type=float,help='Learning rate', default=0.01)
     parser.add_argument('-r','--noise-level',type=float,help='Noise Level', default=0.0)

     args = parser.parse_args()

     X, y = get_data (100, args.noise_level)

     # now train the model 

     # initial guess  

     [w, b] = np.random.random([2])

     train_loss  = [] 
     train_acc = []
     for i in range (0, args.num_iter):
       loss = 0 
       TP, FP, TN, FN = 0, 0, 0, 0 
       for j in range (0, X.shape[0]):
          y_true = y[j]
          ss = sigma (w * X[j] + b)
          y_pred = [ss, 1.0 - ss]
          grad_w, grad_b = grad (w, b, X[j], y[j])
          w -= args.learning_rate * grad_w
          b -= args.learning_rate * grad_b 

          loss += binary_cross_entropy (y_true, y_pred)
          tp, fp, tn, fn = metrics (y_true, ss) 
          TP +=tp
          FP +=fp
          TN +=tn 
          FN +=fn   
                     
       acc = (TP+TN)/(TP+TN+FP+FN) 

       print(i,"loss= %.4f"  %loss,"acc=%.4f" % acc)
       train_acc.append (acc)
       train_loss.append (loss) 

     # now plot the clas and loss  
 
     plot_data (w, b, X, y, train_loss, train_acc)


