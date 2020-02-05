import os
import shutil
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras import optimizers

def one_hot(num_tokens, x):
    x_oh = np.zeros((x.shape[0], x.shape[1], num_tokens))
    for i in range(x.shape[0]):
       for j in range(x.shape[1]):
           x_oh[i, j, x[i,j]] = 1
    return x_oh 

def get_params():
    P = {'num_encoder_tokens': 76,
         'num_decoder_tokens': 77,
         'ndata': 10000, 
         'encoder_width': 60,
         'batch_size': 100,
         'epochs': 4, 
         'validation_split': 0.12, 
         'decoder_width': 40, 
         'latent_dim': 300,
         'output_dir': 'output_lstm'}
    return P 


def clean_dir(P):
    if os.path.isdir(P['output_dir']):
       print("deleting : ", P['output_dir'])
       shutil.rmtree(P['output_dir'])
    if not os.path.exists(P['output_dir']):
       os.makedirs(P['output_dir'])

def get_data(P):
    x = np.random.randint(P['num_encoder_tokens'], size=(P['ndata'], P['encoder_width']))
    y = np.random.randint(P['num_decoder_tokens'], size=(P['ndata'], P['decoder_width']))

    x_train = one_hot(P['num_encoder_tokens'], x)
    y_train = one_hot(P['num_decoder_tokens'], y)

    return x_train, y_train 


def get_model(P):

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, P['num_encoder_tokens']))
    encoder = LSTM(P['latent_dim'], return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, P['num_decoder_tokens']))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(P['latent_dim'], return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
    decoder_dense = Dense(P['num_decoder_tokens'], activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(P['latent_dim'],))
    decoder_state_input_c = Input(shape=(P['latent_dim'],))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model( [decoder_inputs] + decoder_states_inputs,  [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model 

def fit_model(P, model, x_train, y_train):
    # Run training

    encoder_input_data = x_train 
    decoder_input_data = y_train[:,:-1]
    decoder_target_data = y_train[:,1:]

    
    model_checkpoint = ModelCheckpoint(P['output_dir'] + os.sep + 'model.hdf5',
         monitor='val_loss', save_best_only=True, period=1)

    csv_logger = CSVLogger(P['output_dir'] + os.sep + 'history.log')

    tensorboard = TensorBoard(log_dir = P['output_dir'] + os.sep + "tensorboard",
                histogram_freq = 10,
                batch_size = P['batch_size'],
                write_graph = True,
                write_grads = False,
                write_images = False,
                embeddings_freq = 0,
                embeddings_layer_names = None,
                embeddings_metadata = None,
                embeddings_data = None)


    model.compile(optimizer=optimizers.Nadam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    h = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=P['batch_size'],
          epochs=P['epochs'], callbacks= [csv_logger, model_checkpoint, tensorboard],
          validation_split=P['validation_split'])

    return h 

if __name__ == "__main__":

    P = get_params()

    clean_dir(P) 

    model, encoder_model, decoder_model = get_model(P) 
    print(model.summary())

    x_train, y_train = get_data(P)
 
    h = fit_model(P, model, x_train, y_train)
     
    scores = model.evaluate([x_train, y_train[:,:-1]], y_train[:,1:], verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

     
