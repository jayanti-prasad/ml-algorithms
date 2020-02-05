from keras.layers import Flatten, Dropout, Dense, Input, Embedding, LSTM, GRU, BatchNormalization 
from keras.models import Model
import numpy as np
from keras.utils import to_categorical


def get_params():
    P = {'input_vec_len': 10,
         'output_vec_len': 1,
         'num_data': 1000,
         'output_data': 'output_dens'
        }
    return P 

def test_dense(P):

    input_data = Input(shape=(P['input_vec_len'], ), name='Input')
    output_data = Dense(P['output_vec_len'], name='Output') (input_data)
    model = Model(input_data, output_data)

    print("=========Dense==============")
    print(model.summary())


def test_embedding(P):

    input_data = Input(shape=(P['input_vec_len'], ), name='Input')
    x = Embedding(20,P['input_vec_len'], name='Embedding') (input_data)
    output_data = Dense(P['output_vec_len'], name='Output') (x)
    model = Model(input_data, output_data)
    print("=========Embedding=============")
    print(model.summary())


def test_LSTM(P):

    input_data = Input(shape=(P['input_vec_len'], ), name='Input')
    x = Embedding(20,P['input_vec_len'], name='Embedding') (input_data)
    x = LSTM(20, name='LSTM') (x)
    output_data = Dense(P['output_vec_len'], name='Output') (x)
    model = Model(input_data, output_data)
    print("=========LSTM=============")
    print(model.summary())


def test_GRU(P):

    input_data = Input(shape=(P['input_vec_len'], ), name='Input')
    x = Embedding(20,P['input_vec_len'], name='Embedding') (input_data)
    x = GRU(20, name='GRU') (x)
    output_data = Dense(P['output_vec_len'], name='Output') (x)
    model = Model(input_data, output_data)
    print("=========GRU============")
    print(model.summary())


def test_Batch(P):

    input_data = Input(shape=(P['input_vec_len'], ), name='Input')
    x = Embedding(20,P['input_vec_len'], name='Embedding') (input_data)
    x = GRU(20, name='GRU') (x)
    x = BatchNormalization(name='BatchNormalization') (x)
    output_data = Dense(P['output_vec_len'], name='Output') (x)
    model = Model(input_data, output_data)
    print("=========BatchNormalization============")
    print(model.summary())

def test_Drop(P):

    input_data = Input(shape=(P['input_vec_len'], ), name='Input')
    x = Embedding(20,P['input_vec_len'], name='Embedding') (input_data)
    x = GRU(20, name='GRU') (x)
    x = Dropout(0.25, name='Dropout') (x)
    output_data = Dense(P['output_vec_len'], name='Output') (x)
    model = Model(input_data, output_data)
    print("=========Dropout============")
    print(model.summary())

def test_Flat(P):

    input_data = Input(shape=(P['input_vec_len'],1, ), name='Input')
    x = Flatten(name='Flatten') (input_data)
    output_data = Dense(P['output_vec_len'], name='Output') (x)
    model = Model(input_data, output_data)

    print("=========Faltten=============")
    print(model.summary())

 
if __name__ == "__main__":

    P = get_params() 

    test_dense(P) 

    test_embedding(P) 
    
    test_LSTM(P) 
    
    test_GRU(P) 
    test_Batch(P) 
    test_Drop(P) 
    test_Flat(P) 

 
