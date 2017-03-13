from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
import numpy as np

path = os.getcwd()  # reads the current path

def data_set_maker():
    """This function reads .npy files in files folder and returns required data-set for training.
    and final predictions"""

    #  crate a folder in your code directory and name it: "files". put the .npy files iside that folder
    x_test = np.load(path + '/files/tinyX_test.npy', 'r')  # reads the input file

    return x_test


def test_result_file_maker(name, lst):
    """ write the final classification results into a .csv file of our model to evaluate the final performance on test
    set"""
    import csv
    with open(name, 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        file_writer.writerow(['id', 'class'])
        for i in range(len(lst)):
            file_writer.writerow([i, lst[i]])
    return None



X_test = data_set_maker()
X_test = X_test.astype('float32')
X_test /= 255.0

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 64, 64)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(40))
model.add(Activation('softmax'))
model.load_weights(path + '/files/first_try.h5')
pred = model.predict(X_test)
# print(pred)
print(pred[0])
result = []
for lst in pred:
    result.append(np.argmax(lst))

print(result)
print(np.shape(result))
test_result_file_maker("prediction_cnn_1.csv", result)