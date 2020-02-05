import os
import argparse 
import numpy as np
from common_utils import get_utils 
import tensorflow.compat.v1.keras.layers as layers
import tensorflow.compat.v1.keras.models as models
import tensorflow.compat.v1.keras.utils  as utils
import tensorflow.compat.v1.keras.optimizers as optimizers
from  tensorflow.compat.v1.keras.preprocessing import sequence


from keras.datasets import imdb


def get_model (cfg):

    input_data = layers.Input(shape=(cfg.max_len,), name='Input-Layer')
    x = layers.Embedding(cfg.max_features, cfg.latent_dim, name='Embedding') (input_data)
    x = layers.Dropout(0.2, name='Dropout-I')(x)
    x = layers.Conv1D(cfg.filters, cfg.kernel_size, padding='valid', activation='relu', strides=1, name='Conv1D')(x)
    x = layers.GlobalMaxPooling1D(name='Global-Max-Pooling') (x)
    x = layers.Dense(cfg.latent_dim, name='Dense-I') (x)
    x = layers.Dropout(0.2, name='Dropout-II') (x)
    x = layers.Activation('relu', name='Activation-I') (x)

    x  = layers.Dense(1, name='Dense-II')(x)
 
    output_data = layers.Activation('sigmoid', name='Activation-II')(x)

    model = models.Model (input_data, output_data)

    return model 


def fit_model (cfg, model, x_train, y_train, x_test, y_test):

    model_checkpoint, csv_logger, tensorboard = get_utils (cfg)

    model.compile(loss='binary_crossentropy', optimizer='adam',
       metrics=['accuracy'])

    history=model.fit(x_train, y_train,
       batch_size=cfg.batch_size,
       epochs=cfg.num_epochs,
       validation_data=(x_test, y_test),
       callbacks= [csv_logger, model_checkpoint, tensorboard])

    return history 



def get_data(P):
    print('Loading data...')
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=cfg.max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=cfg.max_len)
    x_test = sequence.pad_sequences(x_test, maxlen= cfg.max_len)

    return x_train, y_train, x_test, y_test 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cmod')

    parser.add_argument('-mf', '--max_features',type=int,
        help='Maximum features', default=5000)
    parser.add_argument('-ml', '--max_len',type=int,
        help='Maximum features', default=400)

    parser.add_argument('-bs', '--batch_size',type=int,
        help='Batch Size', default=32)
    parser.add_argument('-ldm', '--latent_dim',type=int,
        help='Embedding dimensions', default=50)
    parser.add_argument('-flt', '--filters',type=int,
        help='Filter size',  default=25)
    parser.add_argument('-krn', '--kernel_size',type=int,
        help='Kernel size', default=3)
    parser.add_argument('-n', '--num_epochs', type=int,
        help='Epochs', default=10)
    parser.add_argument('-w', '--workspace',
        help='Output Directory', required=True)

    cfg = parser.parse_args()

    os.makedirs (cfg.workspace, exist_ok=True)

    model = get_model (cfg)
   
    print(model.summary())
    
    x_train, y_train, x_test, y_test = get_data(cfg)

    h = fit_model (cfg, model, x_train, y_train, x_test, y_test) 


