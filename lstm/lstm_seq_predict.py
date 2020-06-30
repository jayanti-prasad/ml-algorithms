import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  tensorflow.compat.v1.keras.models import Model 
from  tensorflow.compat.v1.keras.layers import Dense,Input,LSTM 

np.random.seed(7)


"""
This is just to show how the next data point can be predicted on the
basis of the past data points in a sequece using a Long-Short-Term-Memory (LSTM)
network. 

For most inputs default values are set and for  explanation you can run the 
program with just '-h' option.


You must provide a data csv file with specifying one column name for
giving the data points.


"""
class SeqModel:
   """
   This is the main LST model. 

   """ 

   def __init__(self, latent_dim, look_back):
        
      inputs = Input (shape=(1, look_back),name="Input-Layer")
      lstm = LSTM(latent_dim,name="LST-Layer") 
      dense = Dense(1)

      x =   lstm (inputs)
      outputs  = dense (x)
      self.model = Model (inputs, outputs)
 

   def train_model (self, trainX, trainY, epochs, batch_size):
      self.model.compile(loss='mean_squared_error', optimizer='adam')
      self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

 
   def predict(self, X):
      return  self.model.predict(X)
  

def normalize (X):
    """
    It is better to map the data between 0 and 1 using 
    the following linear mapping.
    """

    diff = (np.max(X) - np.min(X))
    start = np.min (X)    
    X = (X-start) / diff
    return X, start, diff 


def denormalize (X,start, diff):
    """
    We need to get back to the original scale.

    """
    X = start  + X * diff 
    return X 


def mean_squared_error (y, y_p):
    """
    It can be computed here directly.

    """
    return np.sqrt (np.mean (np.sum ((y-y_p)**2)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--input-file',help='Input CSV file')
    parser.add_argument('-c','--column-name',help='Column Name')
    parser.add_argument('-l','--look-back',type=int,default=1,help='Look back steps')
    parser.add_argument('-d','--latent-dim',type=int,default=4,help='Latent dimension for LSTM')
    parser.add_argument('-n','--num-iter',type=int,default=100,help='Number of iterations')
    parser.add_argument('-b','--batch-size',type=int,default=1,help='Batch Size')
    parser.add_argument('-t','--train-test-split',type=float,default=0.75,help='Train-test split')

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    #X, s, d  = normalize (df[args.column_name])
    X, s, d  = normalize (df['Open Price'])

    # set the target future data after look back days

    Y = X[args.look_back:,]

    # Just for convection add two extra dimensions in the feature data X  
    X = np.expand_dims(X, axis=1)
    X = np.expand_dims(X, axis=2)


    # We must skip the last few data points (look back) since for those
    # we do not have target 

    X = X[:-args.look_back,:,:]


    # get the split point/index for train test split 
    nt = int (X.shape[0] * args.train_test_split)

   
    # now get the train test spli t
    X_train, y_train = X[:nt,:,:],  Y[:nt] 
    X_test, y_test = X[nt:,:,:],  Y[nt:] 
    

    # Now call the model 

    M  =  SeqModel (args.latent_dim, args.look_back)

    # Train the model 
    M.train_model (X_train, y_train, args.num_iter, args.batch_size)

    # Make the prediction for train data 
    y_train_predict = M.predict(X_train)[:,0]

    trainScore = mean_squared_error(y_train, y_train_predict)
    print('Train Score: %.2f RMSE' % (trainScore))

    # Make the prediction for the test data 

    y_test_predict = pd.Series (M.predict(X_test)[:,0])

    # we need this for the plotting purpose 
    y_test_predict.index = df.index[nt:-1]

    testScore = mean_squared_error(y_test, y_test_predict)
    print('Test Score: %.2f RMSE' % (testScore))

    # Now let us plot the predictions 

    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)

    ax.plot(denormalize(Y,s,d),label='Original')

    ax.plot(denormalize(y_train_predict,s,d),'o', label='Train')
    ax.plot(denormalize(y_test_predict,s,d),'o', label='Test')
    plt.legend()

    plt.show()

