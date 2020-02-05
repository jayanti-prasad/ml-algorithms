import sys
import os
import shutil
import numpy as np
from keras.layers import Lambda, Dense, Input, Embedding, BatchNormalization, GRU
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras import optimizers
from ast_data import get_ast_data 

def get_params():
    P = {'num_encoder_tokens': 72,
         'num_decoder_tokens': 72,
         'latent_dim': 300, 
         'data_len': 1000, 
         'encoder_width': 60,
         'decoder_width': 60,
         'batch_size': 400,
         'latent_dim': 300, 
         'epochs': 10,
         'validation_split': 0.12,
         'output_dir':'output_ec'}
    return P 


def get_data(P):
    print("Parameters: ", P )
    X  =  np.random.randint(P['num_encoder_tokens'], 
       size=(P['data_len'], P['encoder_width']))
    Y  =  np.random.randint(P['num_decoder_tokens'], 
       size=(P['data_len'], P['decoder_width']))

    return X, Y 


if __name__ == "__main__":

    P = get_params()

    if os.path.isdir(P['output_dir']):
       print("deleting : ", P['output_dir']) 
       shutil.rmtree(P['output_dir'])
    if not os.path.exists(P['output_dir']):
       os.makedirs(P['output_dir'])

    #X, Y = get_data(P)
    X, Y = get_ast_data(P)


    #encoder 
    encoder_inputs  = Input(shape=(P['encoder_width'],), name='Encoder-Input')

    x = Embedding(P['num_encoder_tokens'], P['latent_dim'], name='Encoder-Embedding', 
        mask_zero=False) (encoder_inputs)

    x = BatchNormalization(name='Encoder-Batchnorm-1')(x)
       
    _, state_h = GRU(P['latent_dim'], return_state=True, name='Encoder-Last-GRU')(x)

    encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')

    encoder_out = encoder_model(encoder_inputs)
      
    #decoder 

    decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

    dec_emb = Embedding(P['num_decoder_tokens'], P['latent_dim'], name='Decoder-Embedding', mask_zero=False)(decoder_inputs)
       
    dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

    decoder_gru = GRU(P['latent_dim'], return_state=True, return_sequences=True, name='Decoder-GRU')

    decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=encoder_out)

    x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)
    decoder_dense = Dense(P['num_decoder_tokens'], activation='softmax', name='Final-Output-Dense')

    decoder_outputs = decoder_dense(x)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
 
    print(model.summary())

    model.compile(optimizer=optimizers.Nadam(lr=0.01),
           loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    encoder_input_data = X 
    decoder_input_data = Y[:, :-1]
    decoder_output_data = Y[:, 1:]

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

    history =   model.fit([encoder_input_data, 
                decoder_input_data],  np.expand_dims(decoder_output_data, -1),
                batch_size = P['batch_size'],
                epochs = P['epochs'], validation_split = P['validation_split'],
                callbacks= [csv_logger, model_checkpoint, tensorboard] )
    
