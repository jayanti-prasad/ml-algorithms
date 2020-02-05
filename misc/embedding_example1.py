from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Input 
from keras.models import Model 
import pandas as pd 
import sys
import re
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import os


def get_params():
   P = {'vocab_size': 1000,
       'vec_len': 2, 
       'latent_dim': 50,
       'epochs': 50, 
       'validation_split': 0.15,
       'output_dir': 'output1' 
       }
   return P 

def clean_text(text):
    words = text.split()
    new_text  = ""
    for w in words:
        if '@' not in w :
            if 'http' not in w:
                new_text = new_text + " " + w 

    new_text =  re.sub('[^A-Za-z0-9]+', ' ', new_text)
    new_text =  re.sub("\d+", " ", new_text)
    return new_text 


def get_twitter_data(): 
    df = pd.read_csv("data/Tweets.csv")
    X = df['text'].tolist() 
    Y = df['airline_sentiment'].tolist()

    D = {'negative': 0, 'neutral': 1,'positive': 2}

    Y = [ D[x] for x in Y]

    X = [clean_text(x) for x in X]

    return X, Y    


if __name__ == "__main__":

    docs, labels  = get_twitter_data()

    for i in range(len(docs)):
        print(i, docs[i], labels[i]) 

    P = get_params()   

     # integer encode the documents
    encoded_docs = [one_hot(d, P['vocab_size']) for d in docs]

    padded_docs = pad_sequences(encoded_docs, maxlen=P['vec_len'], padding='post')

    
    input_data = Input(shape=(P['vec_len'], ), name="Input")
    x = Embedding(P['vocab_size'], P['latent_dim'], name="Embedding") (input_data)
    x = Flatten(name="Flatten")(x)
    output_data = Dense(1,activation='sigmoid') (x)

    model = Model (inputs=input_data, outputs=output_data, name = "Embedding-Model") 

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model

    model_checkpoint = ModelCheckpoint(P['output_dir'] + os.sep + 'model.hdf5',
         monitor='val_loss', save_best_only=True, period=1)

    csv_logger = CSVLogger(P['output_dir'] + os.sep + 'history.log')

    tensorboard = TensorBoard(log_dir = P['output_dir']  + os.sep + 'tensorboard',
        histogram_freq = 10,
        batch_size = 10,
        write_graph = True,
        write_grads = False,
        write_images = False,
        embeddings_freq = 0,
        embeddings_layer_names = None,
        embeddings_metadata = None,
        embeddings_data = None)

    history = model.fit(padded_docs, labels, epochs=P['epochs'], 
        verbose=1, validation_split= P['validation_split'],
        callbacks= [csv_logger, model_checkpoint, tensorboard])
 
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
