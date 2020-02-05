import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split as tts 


def ordinary_least_square (x, y):
    model = linear_model.LinearRegression()
    model.fit(x.reshape(-1,1),y)
    return model 


def ridge_least_square (x, y):
    model = linear_model.Ridge(alpha=.5)
    model.fit(x.reshape(-1,1),y)
    return model


# lasso (least absolute shrinkage and selection operator;
def lasso_least_square (x, y):
    model = linear_model.Lasso(alpha=.5)
    model.fit(x.reshape(-1,1),y)
    return model


def get_data ():
    w = 2.0
    b = 5.0 

    x = np.random.random(100)
    noise = np.random.normal(0.2,0.25,100)
    y = w * x + b
    y = y + noise

    return x, y

    
if __name__ == "__main__":

    X, y = get_data ()


    X_train, X_test, y_train, y_test = tts(X, y, test_size =0.2, random_state=11)

    print(X_train.shape, y_train.shape)  


    models = [ordinary_least_square, ridge_least_square, lasso_least_square]

    for m in models:

       model = m (X_train.reshape(-1,1), y_train)
       print("\n Model:", m.__name__)
       print("Intercept: True 5.0, estimated", model.intercept_)
       print("Coeff: True 2.0, estimated", model.coef_)
