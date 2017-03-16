from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import cv2
import os
from sklearn.utils import shuffle


class EarlyStoppingByLoss(keras.callbacks.Callback):
    def __init__(self, monitor='loss', value=2., verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:
            self.model.stop_training = True



#  reading inputs  ####
def data_set_maker():
    """This function reads .npy files in files folder and returns required data-set for training.
    and final predictions"""

    #  crate a folder in your code directory and name it: "files". put the .npy files iside that folder
    path = os.getcwd()  # reads the current path
    x_train = np.load(path + '/files/tinyX.npy', 'r')  # reads the input file
    y_train = np.load(path + '/files/tinyY.npy', 'r')  # reads the input file
    x_test = np.load(path + '/files/tinyX_test.npy', 'r')  # reads the input file
    x_train, y_train = shuffle(x_train, y_train)

    return x_train, y_train, x_test



def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histo = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram

    cv2.normalize(histo, histo)

    # return the flattened histogram as the feature vector
    return histo.flatten()



#  extract color-histogram feature vectors from input tensor ####
def image_to_feature_vector(raw_tensor):
    """takes a vector of tensors as input and returns the vector of flattened feature vectors as output """
    result = []
    for tensor in raw_tensor:
        result.append(extract_color_histogram(tensor.transpose(2,1,0)))
    return result



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


raw_train_set, y_train, raw_test_set = data_set_maker()
X_train = np.array(image_to_feature_vector(raw_train_set))   # this is our input data-set for training
X_test = np.array(image_to_feature_vector(raw_test_set))   # this is our input data-set for validation
y_train = np_utils.to_categorical(y_train, nb_classes=40)  # encoding labels to one-of-K coding scheme
print(np.shape(X_test))

output_dim = nb_classes = 40
model = Sequential()
model.add(Dense(output_dim, input_dim=len(X_train[0]), activation='softmax'))
batch_size = 128
nb_epoch = 500
model.compile(optimizer=SGD(lr=0.01, momentum=0.90, nesterov=True), loss='categorical_crossentropy')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, mode='auto',
                                              epsilon=0.0001, cooldown=0, min_lr=0)


#  set the value of early_stop callback based on your result of training phase
early_stop = EarlyStoppingByLoss(monitor='loss', value=2.01)

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, callbacks=[reduce_lr, early_stop])



pred = model.predict(X_test)
# print(pred)
print(pred[0])
result = []
for lst in pred:
    result.append(np.argmax(lst))

print(result)
print(np.shape(result))
test_result_file_maker("prediction_logit_1.csv", result)


