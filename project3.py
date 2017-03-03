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
from keras.layers import Activation
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import imutils
from imutils import paths
import argparse
import os


trainX = np.load("/Users/gregorylaredo/Desktop/tinyX.npy") # this should have shape (26344, 3, 64, 64)
trainY = np.load("/Users/gregorylaredo/Desktop/tinyY.npy") 
testX = np.load("/Users/gregorylaredo/Desktop/tinyX_test.npy") # (6600, 3, 64, 64)

# to visualize only
plt.imshow(testX[0].transpose(2,1,0)) # put RGB channels last
plt.show()
#x = trainX[1]
#len(x)
#
#x_reshape = np.reshape(x,[64,64,3])
trainX_reshape = np.reshape(trainX,[len(trainX),64,64,3])



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

# first decided to test implementation on Sample Set of around 1% of images
X_train_sample_set = X_train[0:2600][:][:][:]
y_train_sample_set = y_train[0:2600]
# ^ Note that I checked to seethat all 40 classes are present in sample set

data = []
labels = y_train_sample_set

for (i,X_train_sample_set) in enumerate(X_train_sample_set):
    features = image_to_feature_vector(X_train_sample_set)
    data.append(features)
    
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels)

# going to separate a training and validation set from the sample set (which happened to be part of the training set already separated above)
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.2, random_state=42)

# ======== Keras Package for simple NN
model = Sequential()
model.add(Dense(1536,input_dim=12288,init="uniform",activation="relu"))
model.add(Dense(768,init="uniform",activation="relu"))
model.add(Dense(40))
model.add(Activation("softmax"))

# Training the model via Stochastic Mini-Batch Gradient Descent (SGD)
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128)
print("[INFO] evaluating on test set...")
(loss,accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy*100))

# ======= Predict on Test Set based on above Model
testX_reshape = np.reshape(testX,[len(testX),64,64,3])
testX_data = []
for (i,testX_reshape) in enumerate(testX_reshape):
    features = image_to_feature_vector(testX_reshape)
    testX_data.append(features)
testX_data = np.array(testX_data) / 255.0

pred = model.predict(testX_data, batch_size=128, verbose=1)

pred_cat = np.zeros((len(pred),1))
for i in range(len(pred)):
    temp = pred[i,:]
    pred_cat[i] = np.where(temp == temp.max())
    
# Export to CSV file
np.savetxt("prediction_1%trainingset_kerasNN_2Hlayers.csv",pred_cat)









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
    

    

