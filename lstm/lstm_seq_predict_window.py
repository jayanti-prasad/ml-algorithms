import sys
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
        
      inputs = Input (shape=(look_back,1),name="Input-Layer")
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


def get_frame (X0, look_back):
   dF = pd.DataFrame()
   for i in range(0, look_back):
      col = "X_"+str(i)
      XX =  X0.iloc[i:]
      dF[col] = XX
      dF[col] = dF[col].shift(-i)

   return dF.iloc[:-look_back]


def get_data (args, dF):

    columns  = dF.columns 

    XX = dF[[c for c in columns[:-1]]].copy().to_numpy()
    XX = np.expand_dims(XX, axis=2)

    YY = dF[[columns[-1]]].copy().to_numpy()
    YY = YY.reshape(YY.shape[0])

    # get the split point/index for train test split 
    nt = int (XX.shape[0] * args.train_test_split)

   
    # now get the train test spli t
    X_train, y_train = XX[:nt,:,:],  YY[:nt] 
    X_test, y_test = XX[nt:,:,:],  YY[nt:] 
    
    return X_train, y_train, X_test, y_test, nt


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

    X, s, d  = normalize (df[args.column_name])

    dF = get_frame (X, args.look_back+1)

    X_train, y_train, X_test, y_test, nt = get_data (args, dF)  


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
    y_test_predict.index = dF.index[nt:]

    testScore = mean_squared_error(y_test, y_test_predict)
    print('Test Score: %.2f RMSE' % (testScore))

    # Now let us plot the predictions 

    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)

    ax.plot(denormalize(dF[dF.columns[-1]],s,d),label='Original')

    ax.plot(denormalize(y_train_predict,s,d),'o', label='Train')
    ax.plot(denormalize(y_test_predict,s,d),'o', label='Test')
    plt.legend()

    plt.show()

