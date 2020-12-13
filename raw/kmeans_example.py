import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse 
np.random.seed (seed=292)

"""
This program is a full demo of K-means clustering without using any library.
Comments & Feedback:
- Jayanti Prasad Ph.D [prasad.jayanti@gmail.com]
"""


def get_data ():
    """
    Create two dimensional data for clustering.
    Change the parameters if you wish, I have put two Gaussians.
    """
    cov = [[1, 0], [0, 1]]
    X1 = np.random.multivariate_normal([0,1], cov, 100)
    X2 = np.random.multivariate_normal([4,4], cov, 100)
    X  = np.concatenate((X1, X2), axis=0, out=None)
    df = pd.DataFrame(columns=['x','y'])
    df['x'] = X[:,0]
    df['y'] = X[:,1]
    return df.sample(frac=1)


def get_centeroid (df, num_clusters,columns):
    """
    At every iteration we need to compute the centerioids of the clusters.
    """ 
    centers = np.zeros([num_clusters, len(columns)])
    for i in range(0, num_clusters):
       df1 = df [df['cid'] == float(i)][columns].copy()
       for j in range (0, len(columns)):
          if df1.shape[0] > 0:
             centers[i,j] = df1[columns[j]].mean()
    return centers       


def assign(df, centers):
    """
    At every iteration we assign points to clusters on the basis of their
    proximity.
    """
    cid = np.random.randint(centers.shape[0],size=[df.shape[0]])
    for i in range (0, df.shape[0]):
       dc = np.zeros ([centers.shape[0]])
       for  j in range (0, centers.shape[0]):    
          pos = np.array([df.iloc[i]['x'],df.iloc[i]['y']])
          dc[j] =  np.linalg.norm (pos - centers[j,:])
          cid[i] = np.argmin (dc) 
    df['cid'] = cid
    return df


def get_dispersion(df, centers,columns):
    """
    A measure of the goodness of clustering. You may need to 
    modify it. Here I mostly depend on the visual impression.
    """
    ss = 0.0 
    for i in range (0, centers.shape[0]):
       df1 = df [df['cid'] == float(i)][columns].copy()
       for j in range (0, len(columns)):
          df1[columns[j]] = df1[columns[j]] - centers[i,j]      
          ss += df1[columns[j]].var()
    return ss
 


class Kmeans:
    """
    This the K-mean modulel
    """

    def __init__(self, num_clusters, ndim):
       self.num_clusters = num_clusters 
       self.centers = np.zeros ([num_clusters,ndim])

    def fit (self, df, num_epochs):
       df['cid'] = np.random.randint(int(self.num_clusters),size=[len(df)])
       centers = np.random.randint(df.shape[0]-1,size=[self.num_clusters])
       for i in range(0, self.num_clusters):
           self.centers[i,:] = np.array([df.iloc[i]['x'],df.iloc[i]['y']])     

       for i in range (0, num_epochs):
          df = assign(df, self.centers)
          self.centers = get_centeroid (df, self.num_clusters,['x','y'])
          ss = get_dispersion(df, self.centers,['x','y'])
          print(i, ss)
       return df,ss 

    def predict (self, df):
       return assign(df, self.centers) 
                

def plot_data (df_train,df_test,columns,centers):
 
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(121)
    bx = fig.add_subplot(122)

    ax.set_title('Original data')
    bx.set_title('Clustered data with num cluster= ' + str(centers.shape[0]))

    ax.scatter(df_train['x'],df_train['y'])

    for i in range(0, centers.shape[0]):
        df1 = df_train [df_train['cid'] == float(i)][columns].copy()
        df2 = df_test [df_test['cid'] == float(i)][columns].copy()

        bx.scatter(df1['x'],df1['y'],label='Train [class:' + str(i) +']' )
        bx.scatter(df2['x'],df2['y'],marker='+',label='Test [class:' + str(i) + ']')
        
    bx.scatter(centers[:,0],centers[:,1],c='k',marker="x",label='Centeroids')
    bx.legend()
    plt.show()
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num-epochs',type=int,help='Number of epochs',default=10)
    parser.add_argument('-c','--num-clusters',type=int,help='Number of clusters',default=2)

    args = parser.parse_args()
    df = get_data()

    df_train = df.iloc[:int(0.8*df.shape[0])]
    df_test  = df.iloc[int(0.8*df.shape[0]):]

    print(df_train.shape, df_test.shape) 

    KM = Kmeans (args.num_clusters,df_train.shape[1])
    df_train,ss = KM.fit(df_train,args.num_epochs)

    df_test = KM.predict(df_test)

    plot_data (df_train,df_test,['x','y'],KM.centers)








