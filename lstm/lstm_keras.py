import argparse 
import numpy as np
from tensorflow.compat.v1.keras.layers import LSTM, Dense, Input 
from tensorflow.compat.v1.keras.models import Model 

def num_params (dim_input, dim_output):

    return 4 * (dim_input * dim_output + dim_output *dim_output + dim_output)


if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument('-i','--dim-input',type=int,help='Input dimensions')
   parser.add_argument('-o','--dim-output',type=int,help='output dimensions')

   args = parser.parse_args()

   input_data = Input (shape=([None,args.dim_input]),name='Input-Layer')

   lstm_layer = LSTM (args.dim_output, return_sequences=True, return_state=True,\
     activation="tanh",recurrent_activation="sigmoid",name='LSTM-Layer')

   output_data = lstm_layer (input_data)

   model = Model (input_data, output_data)

   print(model.summary())

   weights = lstm_layer.get_weights()

   U = weights[0]
   W = weights[1]
   B = weights[2]

   #for w in weights:
   #   print(w.shape)
   #print(lstm_layer.weights) 

   #print("U=",U)
   #print("W=",W)
   #print("B=",B)
   print(lstm_layer.weights) 
   print("num params=",num_params (args.dim_input, args.dim_output))

