import os
from keras.layers import Dense, Input 
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

#let us make one even/odd number identifier 

def get_params():
    P={'num_data': 1000,
       'x_dim': 10,
       'y_dim': 1, 
       'epochs': 100,
       'batch_size': 100,
       'validation_split': 0.12,
       'nn_1': 32,
       'nn_2': 64,
       'nn_3': 128,
       'num_tokens_x': 10,
       'num_tokens_y': 2,
       'output_dir': 'output_nn'}

    return P



def get_data (P):

    X = np.random.randint(P['num_tokens_x'], size=(P['num_data'], P['x_dim']))
    #Y = np.random.randint(P['num_tokens_y'], size=(P['num_data'], P['y_dim']))

    Y = [] 
    for i in range(len(X)):
        tt =  np.sqrt(np.sum(X[i]))
        y = 1.0/(1.0+np.exp(-tt))
        print(y)
        Y.append([int(tt %2)]) 


    return X, Y 


if __name__ == "__main__":

 
    P = get_params()

    X, Y = get_data(P)

    #for i in range(len(X)):
    #    print(X[i], Y[i]) 
 
    input_data = Input(shape=(P['x_dim'],), name='Input')

    x = Dense(P['nn_1'], activation='relu', name='Dense-I') (input_data)
    x = Dense(P['nn_2'], activation='relu', name='Dense-II') (x)
    x = Dense(P['nn_3'], activation='relu', name='Dense-III') (x)

    output_data  = Dense(P['num_tokens_y'], activation='softmax', name='Dense-Final') (x)

    model = Model (input_data, output_data)

    print(model.summary())

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  

    Y = to_categorical(Y, P['num_tokens_y'])

    model_checkpoint = ModelCheckpoint(P['output_dir'] + os.sep + 'model.hdf5',
         monitor='val_loss', save_best_only=True, period=1)

    csv_logger = CSVLogger(P['output_dir'] + os.sep + 'history.log')

    tensorboard = TensorBoard(log_dir = P['output_dir']  + os.sep + 'tensorboard',
        histogram_freq = 10,
        batch_size = P['batch_size'],
        write_graph = True,
        write_grads = False,
        write_images = False,
        embeddings_freq = 0,
        embeddings_layer_names = None,
        embeddings_metadata = None,
        embeddings_data = None)

    history = model.fit(X, Y, epochs=P['epochs'],
        verbose=1, validation_split= P['validation_split'],
        callbacks= [csv_logger, model_checkpoint, tensorboard])


