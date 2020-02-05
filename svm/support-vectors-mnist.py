import sys
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from yellowbrick.classifier import ConfusionMatrix
from sklearn.datasets import load_digits

if __name__ == "__main__":

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = tts(X, y, test_size =0.2, random_state=11)

 
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)

    cm = ConfusionMatrix(clf, classes=[0,1,2,3,4,5,6,7,8,9])
    cm.score(X_test, y_test)
    # How did we do?
    cm.show()

