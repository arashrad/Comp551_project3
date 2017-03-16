import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.constraints import maxnorm
from keras import applications
from keras.utils import np_utils
from keras.optimizers import SGD
import os
path = os.getcwd()  # reads the current path


def data_set_maker():
    """This function reads .npy files in files folder and returns required data-set for training.
    It assigns 80 percent of given data_set to training set and the rest to validation set"""

    #  crate a folder in your code directory and name it: "files". put the .npy files iside that folder

    x_all = np.load(path + '/files/tinyX.npy', 'r')  # reads the input file
    y_all = np.load(path + '/files/tinyY.npy', 'r')  # reads the input file

    # split the data into 10% validation-set and 90% training set
    raw_train, raw_valid, y_train, y_valid = train_test_split(x_all, y_all, test_size=0.1, random_state=43)
    return raw_train, raw_valid, y_train, y_valid


def data_set_maker_test():
    """This function reads .npy files in files folder and returns required data-set for training.
    and final predictions"""

    #  crate a folder in your code directory and name it: "files". put the .npy files iside that folder
    x_test = np.load(path + '/tinyX_test.npy', 'r')  # reads the input file

    return x_test


X_train, X_valid, y_train, y_valid = data_set_maker()
Y_train = np_utils.to_categorical(y_train, 40)
Y_valid = np_utils.to_categorical(y_valid, 40)
X_train = np.array(X_train.astype('float32'))
X_valid = np.array(X_valid.astype('float32'))
0


# dimensions of our images.
img_width, img_height = 64, 64

top_model_weights_path = path + '/bottleneck_fc_model_10.h5'
# train_data_dir = 'data/train'
# validation_data_dir = 'data/validation'
nb_train_samples = len(X_train)
nb_validation_samples = len(X_valid)
epochs = 50
batch_size = 16
# print(batch_size)


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


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print model

    def train_gen(X, Y):
        # Y_trash = np.ones(X_test.shape[0])
        flow = datagen.flow(X, Y, batch_size=batch_size, shuffle=False)
        for X1, Y1 in flow:
            yield X1  # ignore Y

    generator = train_gen(X_train, Y_train)
    # datagen.fit(X_train)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    # print bottleneck_features_train
    np.save(open(path + '/bottleneck_features_train1.npy', 'w'),
            bottleneck_features_train)

    generator = train_gen(X_valid, Y_valid)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open(path + '/bottleneck_features_validation1.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open(path + '/bottleneck_features_train1.npy'))
    train_labels = np.array(Y_train)

    validation_data = np.load(open(path + '/bottleneck_features_validation1.npy'))
    validation_labels = np.array(Y_valid)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(4)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', W_constraint=maxnorm(4)))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='softmax'))
    # Compile model
    epochs = 250
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    model.fit(train_data, train_labels, nb_epoch=epochs, batch_size=32,
              validation_data=(validation_data, validation_labels), shuffle=False)
    model.save_weights(top_model_weights_path)

)


# save_bottlebeck_features()
train_top_model()
