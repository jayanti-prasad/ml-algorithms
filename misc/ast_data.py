import os
import pandas as pd
from ktext.preprocess import processor
import dill as dpickle

def get_ast_data(P):
    df = pd.read_csv("data/training.csv")

    encoder_pp = processor(keep_n = P['num_encoder_tokens']-2,
        padding_maxlen = P['encoder_width'])

    decoder_pp = processor(append_indicators = True, keep_n = P['num_decoder_tokens']-2,
        padding_maxlen = P['decoder_width'], padding ='post')

    X = df['prev'].tolist()
    Y = df['curr'].tolist()

    x_train  = encoder_pp.fit_transform(X)
    y_train  = decoder_pp.fit_transform(Y)

    with open(P['output_dir'] + os.sep + 'encoder_pp.dpkl', 'wb') as f:
        dpickle.dump(encoder_pp, f)

    with open(P['output_dir'] + os.sep + 'decoder_pp.dpkl', 'wb') as f:
        dpickle.dump(decoder_pp, f)

    return x_train, y_train 

if __name__ == "__main__":

    P = {'num_encoder_tokens': 72,
         'num_decoder_tokens': 72, 
         'encoder_width': 60,
         'decoder_width': 60}

    x_train, y_train = get_ast_data(P)

    print(x_train.shape)
    print(y_train.shape)
