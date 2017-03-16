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


class LossHistoryTrain(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class LossHistoryValid(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))


class AccHistoryTrain(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('acc'))


class AccHistoryValid(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('val_acc'))


#  reading inputs  ####
def data_set_maker():
    """This function reads .npy files in files folder and returns required data-set for training.
    It assigns 80 percent of given data_set to training set and the rest to validation set"""

    #  crate a folder in your code directory and name it: "files". put the .npy files iside that folder
    path = os.getcwd()  # reads the current path
    x_all = np.load(path + '/files/tinyX.npy', 'r')  # reads the input file
    y_all = np.load(path + '/files/tinyY.npy', 'r')  # reads the input file

    # split the data into 10% validation-set and 90% training set
    raw_train, raw_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.2, random_state=43)
    return raw_train, raw_valid, y_train, y_valid



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


raw_train_set, raw_valid_set, y_train, y_valid = data_set_maker()
X_train = np.array(image_to_feature_vector(raw_train_set))   # this is our input data-set for training
X_valid = np.array(image_to_feature_vector(raw_valid_set))   # this is our input data-set for validation
y_train = np_utils.to_categorical(y_train, nb_classes=40)  # encoding labels to one-of-K coding scheme
y_valid = np_utils.to_categorical(y_valid, nb_classes=40)  # encoding labels to one-of-K coding scheme


output_dim = nb_classes = 40
model = Sequential()
model.add(Dense(output_dim, input_dim=len(X_train[0]), activation='softmax'))
batch_size = 128
nb_epoch = 2000
data = []
path = os.getcwd()
model.compile(optimizer=SGD(lr=0.01, momentum=0.90, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

t_loss_history = LossHistoryTrain()
v_loss_history = LossHistoryValid()
t_accuracy_history = AccHistoryTrain()
v_accuracy_history = AccHistoryValid()

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=0, mode='auto',
                                              epsilon=0.0001, cooldown=0, min_lr=0)

csv_logger = keras.callbacks.CSVLogger("training_histogram.log", separator=',', append=False)


model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_valid, y_valid),
          callbacks=[t_loss_history, t_accuracy_history, v_accuracy_history, v_loss_history, reduce_lr, csv_logger])


# print t_accuracy_history.accuracy
# print v_accuracy_history.accuracy

xt = range(len(t_loss_history.losses))
xv = range(len(v_loss_history.losses))
ta = t_accuracy_history.accuracy
va = v_accuracy_history.accuracy
tl = t_loss_history.losses
vl = v_loss_history.losses




fig, ax = plt.subplots()
line_1, = ax.plot(tl, label='Inline label')
line_2, = ax.plot(vl, 'r-', label='Inline label')
# Overwrite the label by calling the method.
line_1.set_label('training set')
line_2.set_label('validation set')
ax.legend()
ax.set_ylabel('Loss')
ax.set_title('')
ax.set_xlabel('epoch')
plt.show()
