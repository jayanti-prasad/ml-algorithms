import numpy as np

"""
This program implements a decision tree algorithm from scratch 
using only numpy.

The data used is also created here, basically we create a data set 
with 100 rows and 5 columns. The values for the first four columns (features)
are drawn from a uniform random generator.
In order to get the target columns we do the following:
- add the values of all the features and call that y0
- now compute the mean of y0
- for all the rows if y0 < mean set y = 0 else y = 1
There is nothing special about this data. Once can create or use
any data the only condition is that the target column should be only 0 and 1 

This is only a wireframe model and for the actual case we need to add 
a lot of meat.

For comments & suggestions : prasad.jayanti@gmail.com 
- Jayanti Prasad Ph.D

"""

def get_data ():
   # create some data with binary 0,1 outcomes 

   np.random.seed (seed=765)
   X = np.random.random([100,4])
   y = np.sum(X, axis=1)
   mu = y.mean()
   y = [0 if  a < mu else 1 for a in y]
   return X, y


def get_gini (y):
   # compute gini index  
   class_label = list (set (y))
   m = len (y)
   y = np.array(y)
   group_count  = [np.sum(y == c) for c in class_label]
   gini = 1.0 - sum((n / m) ** 2 for n in group_count)
   return gini


def best_split (X, y):
    # find out for which of the features we have a best split 
    # and what is the gini index and threshold for that. 

    # get the elements of the class 
    y_class  = list (set (y))
    # number of classes 
    num_class = len (y_class)
    num_features = X.shape[1] 

    # For evert feature we want to find out the best threshold & gini index
    # corresponds to that.
    
    G = []
    for i in range (0, num_features):
       # This is the maximum gini index we can have
       # we want to find the minimum
       g_min = 1.0
       threshold = 0.0 
       # now load the feature column 
       xx = X[:, i]

       # sort its values 
       thr = np.sort(xx)
       # find out a list of thresholds from the mid-points
       
       tr = [ (thr[i] + thr[i+1])/2 for i in range (0, thr.shape[0]-1)]

       # now iterate over all the thresholds  
       for t in tr:
          # get in index of the data points for the left
          
          idx_l  = [i for i in range (0, len (xx)) if xx[i] < t]
          # the rest will go to right 
          idx_r  = [i for i in range (0, len (xx)) if i not in idx_l]

          # get the actual data which goes to left & right 
          yy_l = [y[i] for i in idx_l]
          yy_r = [y[i] for i in idx_r]

          # compute the gini index for the left & right split  
          g_l = get_gini (yy_l)
          g_r = get_gini (yy_r)

          # take the weighted average of the left & right gini
          
          g = g_l * len(yy_l) / len (y) + g_r * len (yy_r) / len (y)

          # check if we have found a lower gini index for a feature 
          if g < g_min:
             g_min = g
             threshold = t
       D = {'feature': i, "threshold": t, "gini": g_min}
       G.append (D)

    g = [a['gini'] for a in G] 
    c_split = np.argmin(g)
    return G [c_split] 



class Node:
   # This holds the nodes of our binary decision tree  

   def __init__(self, data):
      # indices of all the data points associated with 
      # this node 
      self.data = data
      # left node 
      self.left = None
      # right node  
      self.right = None
      # id of the node  
      self.id =  None 
      # data for the split such as feture used, threshold & gini index 
      self.split = {}
    

class BTree:
   # This is the actual (binary) decision tree 
   # more methods need to be added 
   def __init__(self, X, y):
      self.rec = 0      
      self.num_nodes = 0 
      self.count =0 
      self.X = X
      self.y = y 
      D = [i for i in range (0, len(y))]
      self.root = Node (D)

      # Now we can apply 'build' method on the tree which
      # takes two arguments - the parent and the indices of the
      # data point we want to associate with it.
      self.__build__(self.root, D)


   def __build__ (self, parent, data):

      # data is a list which has ids of all the 
      # data points assiciated with a node 

      node = Node (data)
    
      if not self.root:
          self.root = node 

      # now get the actual feature & label data 
      X = np.array([self.X[i,:] for i in data])
      y = [self.y[i] for i in data]

      # find out the best split - the feature & its threshold  
      split  = best_split (X, y)
      print("split:",split)
      # dump the split to node data 
      parent.split = split 
    
      # now split the data points (indices) on the basis of the 
      # decsion about the best feature 
      data_l  = [i for i in data  if self.X[i,split['feature']] < split['threshold']]

      # what does not go to lest goes to right 
      data_r  = [i for i in data  if i not in data_l]

      # now add lesft & right nodes to the parent 
      parent.left = Node (data_l)
      parent.left.id = self.num_nodes 
      self.num_nodes +=1 

      parent.right = Node (data_r)
      parent.right.id = self.num_nodes 
      self.num_nodes +=1 

      # check if
      y_l = [self.y[i] for i in data_l]
      y_r = [self.y[i] for i in data_r] 

      # iterate if left & right groups are not homogeneous   
      if get_gini (y_l) > 0.0:
         self.__build__ (parent.left, data_l)
      if get_gini (y_r) > 0.0 :
         self.__build__ (parent.right, data_r)
      
   def traverse (self, node ):
      # This method is just to show how to traverse the tree
      # and servers a template for complex tree operations   

      labels = [self.y[i] for i in  node.data] 
      print("data:",node.id, node.split, labels)
      if  node.left:
         self.traverse(node.left)
      if  node.right:
         self.traverse(node.right)

   def predict (self, node, X):
      # This is basically binary search tree 
      # model and will find out which of the leaf
      # a test data point should go
      # the label if decided by the median of the 
      # the target lead found 
      # this may need more polishin 
      split = node.split
      y = [self.y[i] for i in node.data] 
      if split['gini'] == 0.0: 
         print("target node:",node.id, "predicted class:", int(np.median(y))) 
      if 'gini' in split :
         if  split['gini'] > 0.0 :
            if X[split['feature']] < split['threshold']:
               next_node = node.left 
            else :
               next_node = node.right 
            self.predict (next_node, X)
 

if __name__ == "__main__":
   # This is the main calling program 
  
   X, y = get_data () 

   # Load the data in a binary decision tree 
   T = BTree (X,y)

   # We can traverse the tree to know what all the data our nodes have 
   T.traverse (T.root)
  
   node = T.predict (T.root, X[10,:])
   print("true class:", y[10])  
