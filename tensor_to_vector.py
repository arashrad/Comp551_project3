from sklearn.model_selection import train_test_split
import scipy.io
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# crate a folder in your code directory and name it: "files". put the .npy files iside that folder 
path = os.getcwd()  # reads the current path
x_all = np.load(path + '/files/tinyX.npy', 'r')  # reads the input file
y_all = np.load(path + '/files/tinyY.npy', 'r')  # reads the input file

# split the data into 10% validation-set and 90% training set
raw_train, raw_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.2, random_state=42)




def vectorizer(tensor):
    """takes a single matrix and return the corresponding vector of it"""
    result = []
    for matrix in tensor:
        for row in matrix:
            for item in row:
                result.append(item)
    return result


X_train = []
counter = 0
for tensor in raw_train:
    print(counter)
    X_train.append(vectorizer(tensor))
    counter += 1
