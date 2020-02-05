import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import argparse 
import tensorflow.compat.v1.keras.layers as layers
import tensorflow.compat.v1.keras.models as models
import tensorflow.compat.v1.keras.optimizers as optimizers 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from common_utils import get_utils


def get_model(cfg):

    input_data = layers.Input(shape=(4,), name='Input')
    x = layers.Dense(30, activation='relu', name='Dense-I') (input_data)
    x = layers.Dense(10,activation='relu', name='Dense-II') (x)
    x = layers.Dense(5,activation='relu', name='Dense-III') (x)
    output_data  = layers.Dense(3,activation='softmax', name='Dense-IV') (x)
    model = models.Model(input_data, output_data)

    return model 


def fit_model (cfg, x_train, y_train, x_test, y_test):
   
    model.compile(optimizers.Adam(lr=0.04),'categorical_crossentropy', metrics=['accuracy'])

    model_checkpoint, csv_logger, tensorboard = get_utils (cfg)

    #fitting the model and predicting 
    history = model.fit(x_train, y_train, epochs=cfg.num_epochs, validation_data=(x_test, y_test),
        callbacks= [csv_logger, model_checkpoint, tensorboard])

    return history 
 

def get_data(P):

    iris = datasets.load_iris()
    X = iris.data[:, 0:4]  # we only take the first two features.
    y = iris.target

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y =  pd.get_dummies(y1).values


    x_train, y_train, x_test, y_test \
       = train_test_split(X,Y,test_size=cfg.test_split,random_state=cfg.random_seed)
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

    cfg = parser.parse_args()

    os.makedirs (cfg.workspace, exist_ok=True)

    model = get_model (cfg)

    print(model.summary())

    x_train, x_test, y_train, y_test = get_data(cfg) 

    print(x_train.shape)
    print(y_train.shape)

    h =  fit_model (cfg, x_train, y_train, x_test, y_test) 

    y_pred = model.predict(x_test)

    y_test_class = np.argmax(y_test,axis=1) 
    # convert encoded labels into classes: say [0, 0, 1] -->  [2] i.e Iris-virginica
    y_pred_class = np.argmax(y_pred,axis=1) 
    # convert predicted labels into classes: say [0.00023, 0.923, 0.031] -->  [1] i.e. Iris-versicolor

    #Accuracy of the predicted values
    print(classification_report(y_test_class, y_pred_class)) # Precision , Recall, F1-Score & Support
    cm = confusion_matrix(y_test_class, y_pred_class)
    print(cm)
    # visualize the confusion matrix in a heat map
    df_cm = pd.DataFrame(cm)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
