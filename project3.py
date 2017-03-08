"""
Created on Wed Mar  1 12:50:47 2017

======= COMP 551: Project 3 - Image Classification ========

@author: gregorylaredo
"""

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import skimage.color
#import skimage.feature
#import theano
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, pooling, Reshape, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import imutils
from imutils import paths
import argparse
import os


trainX = np.load("/Users/gregorylaredo/Desktop/tinyX.npy") # this should have shape (26344, 3, 64, 64)
trainY = np.load("/Users/gregorylaredo/Desktop/tinyY.npy") 
testX = np.load("/Users/gregorylaredo/Desktop/tinyX_test.npy") # (6600, 3, 64, 64)

# to visualize only
plt.imshow(testX[10].transpose(2,1,0)) # put RGB channels last
plt.show()
#x = trainX[1]
#len(x)
#
#x_reshape = np.reshape(x,[64,64,3])
trainX_reshape = np.reshape(trainX,[len(trainX),64,64,3])


# Probability of each Class
cat,count = np.unique(trainY,return_counts=True)
prob = count/len(trainY)



# ========== Feature Matrix: Generate a linear vector for each
def image_to_feature_vector(image,size=(64,64)):
    return cv2.resize(image,size).flatten()

#Grey_scale_1 = skimage.color.rgb2grey(x_reshape)
#SIFT,SIFT_image = skimage.feature.daisy(Grey_scale_1,visualize=True)
#np.shape(SIFT)
#
#plt.imshow(SIFT_image)
#plt.show()




# ========== Randomly shuffling the data and splitting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(trainX_reshape, trainY, test_size=0.2, random_state=42)

## first decided to test implementation on Sample Set of around 1% of images
#X_train_sample_set = X_train[0:10000][:][:][:]
#y_train_sample_set = y_train[0:10000]
## ^ Note that I checked to seethat all 40 classes are present in sample set

data_train = []
#labels = y_train

for (i,X_train) in enumerate(X_train):
    features = image_to_feature_vector(X_train)
    data_train.append(features)
data_train = np.array(data_train) / 255.0
#labels = np_utils.to_categorical(labels)

data_val = []
for (i,X_val) in enumerate(X_val):
    features = image_to_feature_vector(X_val)
    data_val.append(features)
data_val = np.array(data_val) / 255.0

## going to separate a training and validation set from the sample set (which happened to be part of the training set already separated above)
#(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainX_reshape_random, x, trainY_random, y = train_test_split(trainX_reshape, trainY, test_size=0.0, random_state=42)
data_ALL_train = []
for (i,trainX_reshape_random) in enumerate(trainX_reshape_random):
    features = image_to_feature_vector(trainX_reshape_random)
    data_ALL_train.append(features)
data_ALL_train = np.array(data_ALL_train) / 255.0
y_ALL_train = trainY_random

# ======== Keras Package for simple NN
model = Sequential()
model.add(Dense(output_dim=512,input_dim=12288,activation="relu")) # Hidden Layer 1
model.add(Dense(output_dim=512,activation="relu")) # HL 2
model.add(Dense(output_dim=512,activation="relu")) # HL 3
model.add(Dense(output_dim=512,activation="relu")) # HL 4
model.add(Dense(output_dim=256,activation="relu")) # HL 5
model.add(Dense(output_dim=40))
model.add(Activation("softmax"))

# Training the model via Stochastic Mini-Batch Gradient Descent (SGD)
print("[INFO] compiling model...")
#sgd = SGD(lr=0.01)
BS = 256 # batch size
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=0.001),
              metrics=["accuracy"])
history = model.fit(data_ALL_train, y_ALL_train, 
                    validation_split=0.20, 
                    nb_epoch=20, batch_size=BS)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# Can't do below when fitting to all data and already testing on a validation set in the .fit code
print("[INFO] evaluating on test set...")
(loss,accuracy) = model.evaluate(data_val, y_val, batch_size=BS, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy*100))


# ======== Keras Package for Convolutional NN
model_CNN = Sequential()
model_CNN.add(Reshape((3,64,64), input_shape=(12288,))) # HL 1
model_CNN.add(Conv2D(64,3,3, border_mode='same', subsample=(4,4), activation='relu')) # HL 2
model_CNN.add(pooling.AveragePooling2D(border_mode='same')) # HL 3
model_CNN.add(Conv2D(32,3,3, border_mode='same', activation='relu'))
#model_CNN.add(BatchNormalization())
#model_CNN.add(Dropout(0.5))
model_CNN.add(Flatten())
model_CNN.add(Dense(output_dim=256, activation='relu')) # HL 4
#model_CNN.add(BatchNormalization())
model_CNN.add(Dropout(0.5))
model_CNN.add(Dense(output_dim=40, activation='softmax'))
print("[INFO] compiling model...")
BS = 256
model_CNN.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])
history = model_CNN.fit(data_ALL_train, y_ALL_train, 
              validation_split=0.20, 
              nb_epoch=50, batch_size=BS)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()


# ======= Predict on left-over of original training set based on above Model
X_to_predict = X_train[15001:21075][:][:][:]
y_to_predict = y_train[15001:21075]
data_to_predict = []
for (i,X_to_predict) in enumerate(X_to_predict):
    features = image_to_feature_vector(X_to_predict)
    data_to_predict.append(features)
data_to_predict = np.array(data_to_predict) / 255.0

pred = model_CNN.predict(data_to_predict, batch_size=BS, verbose=1)
pred_cat = np.zeros((len(pred),1))
for i in range(len(pred)):
    temp = pred[i,:]
    pred_cat[i] = np.where(temp == temp.max())

y_pred = pred_cat    
y_true = y_to_predict 
from sklearn.metrics import metrics
print (metrics.classification_report(y_true, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
plt.matshow(cm)


# ======= Predict on Test Set based on above Model
testX_reshape = np.reshape(testX,[len(testX),64,64,3])
testX_data = []
for (i,testX_reshape) in enumerate(testX_reshape):
    features = image_to_feature_vector(testX_reshape)
    testX_data.append(features)
testX_data = np.array(testX_data) / 255.0

pred = model.predict(testX_data, batch_size=BS, verbose=1)

pred_cat = np.zeros((len(pred),1))
for i in range(len(pred)):
    temp = pred[i,:]
    pred_cat[i] = np.where(temp == temp.max())
    
# Export to CSV file
np.savetxt("prediction_15000trainingset_kerasCNN_3Hlayers_f1score.csv",pred_cat)









# ======= Fully-connected Feedforward NN: ========
# =========== 1 Hidden Layer to start with

N = len(X_train)
input_dim = len(np.transpose(X_train)) # number of Features = number of Inputs to NN
output_dim = 40 # We have 40 casses of images possible

# Gradient descent parameters (chosen by hand)
alpha = 0.01 # learning rate
lambda_reg = 0.01 # regularisation strength

# Calculating Loss Function
def Loss(model):
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']
    # Forward Propagation to calculate predictions
    input_layer1 = X_train.dot(W1) + b1
    output_layer1 = np.tanh(input_layer1)
    input_layer2 = X_train.dot(W2) + b2
    exp_scores = np.exp(input_layer2)
    probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
    # Calculating Loss
    logprobs = -np.log(probs[range(N),y_train])
    data_loss = np.sum(logprobs)
    # Adding Regularization term to Loss
    data_loss += lambda_reg/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./N * data_loss
    

    

