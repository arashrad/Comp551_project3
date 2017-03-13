# first create a folder in your current directory and name it as "files" then put all your .npy files inside that


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


path = os.getcwd()  # reads the current path
def data_set_maker():
    """This function reads .npy files in files folder and returns required data-set for training.
    It assigns 80 percent of given data_set to training set and the rest to validation set"""

    #  crate a folder in your code directory and name it: "files". put the .npy files iside that folder

    x_all = np.load(path + '/files/tinyX.npy', 'r')  # reads the input file
    y_all = np.load(path + '/files/tinyY.npy', 'r')  # reads the input file

    # split the data into 10% validation-set and 90% training set
    raw_train, raw_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.2, random_state=43)
    return raw_train, raw_valid, y_train, y_valid


X_train, X_valid, y_train, y_valid = data_set_maker()
Y_train = np_utils.to_categorical(y_train, 40)  # convert to categorical, one of K vector
Y_valid = np_utils.to_categorical(y_valid, 40)  # convert to categorical, one of K vector
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_valid = X_valid / 255.0  # normalizing validation-set

#  creating the model
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
nb_epoch = 50
lrate = 0.01
decay = lrate/nb_epoch
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# using keras ImageDataGenerator class for data augmentation
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    samples_per_epoch=len(X_train), nb_epoch=nb_epoch, validation_data=(X_valid, Y_valid))
model.save_weights(path + '/files/first_try_1.h5')  # saving the model's final weights into the files folder
# after training the model you can use cnn_prediction script to use current model to  predict the test-dataset based
# on the saved weights.





