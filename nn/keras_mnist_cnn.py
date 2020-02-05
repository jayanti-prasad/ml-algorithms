import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import tensorflow.compat.v1.keras.layers as layers
import tensorflow.compat.v1.keras.utils as utils 
import tensorflow.compat.v1.keras.models as models
import tensorflow.compat.v1.keras.optimizers as optimizers
import tensorflow.compat.v1.keras.losses as losses 
import tensorflow.compat.v1.keras.datasets as datasets 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from common_utils import get_utils

img_rows, img_cols  = 28, 28 


def get_model (cfg):

    input_shape = (img_rows, img_cols, 1)

    
    input_data = layers.Input(shape=input_shape,  name = "Input-Layer")

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='Conv2D-I') (input_data)
    x = layers.Conv2D(64, (3, 3), activation='relu',name='Conv2D-II') (x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool') (x)
    x = layers.Dropout(0.25, name='Dropout-I') (x)
    x = layers.Flatten(name='Flatten') (x)
    x = layers.Dense(128, activation='relu', name='Output-Dense') (x)
    x = layers.Dropout(0.5, name='Dropout-II') (x)

    output_data = layers.Dense(cfg.num_classes, activation='softmax') (x)

    model = models.Model (inputs=input_data, outputs=output_data, name='Conv2D-Model')

    return model 


def fit_model (cfg, model, x_train, y_train, x_test, y_test):

    model_checkpoint, csv_logger, tensorboard = get_utils (cfg)

    model.compile(loss=losses.sparse_categorical_crossentropy,
       optimizer=optimizers.Adadelta(),
       metrics=['accuracy'])

    history = model.fit(x_train, y_train,
          batch_size=cfg.batch_size,
          epochs=cfg.num_epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks= [csv_logger, model_checkpoint, tensorboard])



def get_data(P):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

 
    #for x in [x_train, x_test]:
    x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    #for x in [x_train, x_test]:
    x_test = x_test.reshape(x_test.shape[0],  img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    

    print('x_train shape:', x_train.shape)

    for y in [y_train, y_test]:
         # convert class vectors to binary class matrices
         y = utils.to_categorical(y, cfg.num_classes)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='cmod')

    parser.add_argument('-rs', '--random_seed',type=int,
        help='Maximum features', default=657)
    parser.add_argument('-bs', '--batch_size',type=int,
        help='Batch Size', default=32)
    parser.add_argument('-ts', '--test_split',type=float,
        help='Batch Size', default=0.2)
    parser.add_argument('-n', '--num_epochs', type=int,
        help='Epochs', default=10)
    parser.add_argument('-w', '--workspace',
        help='Output Directory', required=True)

    parser.add_argument('-ncl', '--num_classes', type=int,
        help='Epochs', default=10)

    cfg = parser.parse_args()

    os.makedirs (cfg.workspace, exist_ok=True)

    model = get_model (cfg)

    print(model.summary())
 
    x_train, y_train, x_test, y_test   = get_data(cfg)  

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)


    model = get_model (cfg)


    print(model.summary())

    #sys.exit()

    h = fit_model (cfg, model,  x_train, y_train, x_test, y_test)


    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
