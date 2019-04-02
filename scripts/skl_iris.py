# A simple python script using the scikit-learn package
# for machine learning w/ K-Nearest-Neighbors method taken from
# https://www.geeksforgeeks.org/introduction-machine-learning-using-python/

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = load_iris() # dataset of iris characteristics
# data['data'] contains 150 data sets of:
## arrays of 4 feature data points, one set per sample
# data['target'] contains 150:
## 0, 1, or 2 identities for the above data
# data['target_names'] contains 3:
## 'setosa', 'versicolor', 'virginica' names of 0, 1, and 2
# data['DESCR'] contains:
## long description of dataset
# data['feature_names'] contains:
## string names of features (sepal length/width, petal length/width)
# data['filename'] contains:
## string path and filename for .csv dataset

x_train, x_test, y_train, y_test = train_test_split(
        data["data"], data["target"], random_state=0)  # grab train and test data
# x is feature (eg petal length), y is target (eg 0 [setosa])
# data is "randomly" split 75:25 into train:test data

kn = KNeighborsClassifier(n_neighbors=1) # only consider one nearest neighbor
kn.fit(x_train, y_train) # fit the ML object w/ training data

x_new = np.array([[5,2.9,1,0.2]]) # an unknown iris
prediction = kn.predict(x_new) # predict its identity based on its nearest neighbor

print("Predicted target value: {}\n".format(prediction))
print("Predicted feature name: {}\n".format
            (data["target_names"][prediction])) 
print("Test score: {:.2f}".format(kn.score(x_test, y_test))) 
# test the score: no. of predictions correct / total predictions

