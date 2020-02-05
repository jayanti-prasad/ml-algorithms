"""------------------------------------------------------------------------
1) This program demonstrate the use of Decision Tree classifier using 
scikit-learen

2) There are two example data sets used :
 - Iris data set 

3) Output of the program is prediction for a set of test examples, accuracy and 
decision trees.

-  Jayanti Prasad [prasad.jayanti@gmail.com]
   October 30, 2019 
---------------------------------------------------------------------------"""
import sklearn
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import sklearn.datasets as datasets
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

def get_iris_data():
    iris = sklearn.datasets.load_iris()
    X = iris['data']
    y = iris['target']
    feature_cols = iris.feature_names
    target_cols = iris.target_names
    return X, y, feature_cols, target_cols 


if __name__ == "__main__":

    # get the iris data 
    X, y, feature_cols, target_cols = get_iris_data()

    # split the data  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # get the classifier 
    clf = DecisionTreeClassifier()

    # train the classifier 
    clf = clf.fit(X_train,y_train)

    # make prediction 
    y_pred = clf.predict(X_test)

    # get the accuracy 
    acc = metrics.accuracy_score(y_test, y_pred) 
    print("Accuracy:",acc)

    # plot the tree 
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=target_cols)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    graph.write_png('iris.png')
    Image(graph.create_png())
   


