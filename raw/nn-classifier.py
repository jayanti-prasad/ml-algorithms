import numpy as np
from tensorflow.compat.v1.keras.layers import Input, Dense 
from  tensorflow.compat.v1.keras.models import Model
from  tensorflow.compat.v1.keras.utils import to_categorical 
from  tensorflow.compat.v1.keras.optimizers import SGD, Adam, Adamax, RMSprop, Adagrad, Adadelta, Nadam 
from sklearn.model_selection import train_test_split 


def get_data ():

    n, p = 1000, 10
    X = np.random.randint (100, size=(n, p))
    #y = np.random.randint (2, size=(n, 1))
    y = np.sum(X, axis=1)
    y = [ x % 2 for x in y] 
    y = to_categorical(y, 2)

    return X, y


def get_model ():

    input_data = Input(shape=(10,), name='Input')

    hidden_1 = Dense (15, activation='relu', name='hidden_1')
    hidden_2 = Dense (25, activation='relu', name='hidden_2')
    output  =  Dense (2, activation='softmax', name='output')

    x = hidden_1 (input_data)
    x = hidden_2 (x)
    output_data  = output (x)

    model = Model (input_data, output_data)
    return model 


if __name__ == "__main__":

    X, y  = get_data ()

    #for  i in range (0, 1000):
    #   print(X[i], y[i])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1343)

    model = get_model ()

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    #optimizer = Adagrad(lr=0.01)
    #optimizer = Adadelta(lr=1.0, rho=0.95)
    #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) 
    #optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
    #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
   
    #optimizer = RMSprop(learning_rate=0.001, rho=0.9)
    #optimizer = Adam ()

    model.compile (optimizer=optimizer, loss='binary_crossentropy',  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_split=0.1)



    print(model.summary())

