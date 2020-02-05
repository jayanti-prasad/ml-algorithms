import os
import sys
import numpy as np
from keras.layers import Input, Dense, Embedding, BatchNormalization, Flatten
from keras.models import Model 
from keras.preprocessing.text import one_hot
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

def get_data(P):

    X = np.random.randint(P['num_tokens_x'], 
        size=(P['num_data'], P['x_len']))
   
    Y = np.zeros([P['num_data'], 1], dtype=int)
    for i in range(P['num_data']):
        tt = np.sum(X[i]) % 2 
        Y[i, 0] =  int(tt)  
    
    return X, Y

if __name__ == "__main__":
     
    P = {'num_data':1000, 
         'x_len': 20, 
         'y_len': 1, 
         'num_tokens_x': 100, 
         'num_tokens_y': 10,
         'latent_dim': 50,
         'output_dir': 'output'}

    X, Y = get_data(P)

    input_data = Input(shape=(P['x_len'], ), name='Input-Layer')
    x = Embedding(P['num_tokens_x'], P['latent_dim'], name='Embedding-Layer') (input_data)
    x = Flatten(name='Batch-Normalization') (x)

    output_data  = Dense(P['y_len'], activation='sigmoid', name='Output-Dense') (x)     

    model = Model(inputs=input_data, outputs=output_data, name='Embedding-Model') 

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) 

    print(model.summary())

    model_checkpoint = ModelCheckpoint(P['output_dir'] + os.sep + 'model.hdf5', 
         monitor='val_loss', save_best_only=True, period=1)

    csv_logger = CSVLogger(P['output_dir'] + os.sep + 'history.log')

    tensorboard = TensorBoard(log_dir = P['output_dir'] + os.sep + 'tensorboard',
                              histogram_freq = 10,
                              batch_size = 10,
                              write_graph = True,
                              write_grads = False,
                              write_images = False,
                              embeddings_freq = 0,
                              embeddings_layer_names = None,
                              embeddings_metadata = None,
                              embeddings_data = None)

    history = model.fit(X, Y, batch_size=10, epochs=20, validation_split=0.12,  
        callbacks= [csv_logger, model_checkpoint, tensorboard]) 
  
    X_test = np.random.randint(P['num_tokens_x'], size=(1, P['x_len']))

    print(X_test)
  
    Y_test = model.predict(X_test)
    print(Y_test)
