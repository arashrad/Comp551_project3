"""
Created on Wed Mar  1 12:50:47 2017

======= COMP 551: Project 3 - Image Classification ========

@author: gregorylaredo
"""

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


trainX = np.load("/Users/gregorylaredo/Desktop/tinyX.npy") # this should have shape (26344, 3, 64, 64)
trainY = np.load("/Users/gregorylaredo/Desktop/tinyY.npy") 
testX = np.load("/Users/gregorylaredo/Desktop/tinyX_test.npy") # (6600, 3, 64, 64)

# to visualize only
plt.imshow(trainX[1].transpose(2,1,0)) # put RGB channels last
plt.show()
x = trainX[1][1]
len(x)

# Randomly shuffling the data
raw_train, raw_val, y_train, y_val = train_test_split(trainX, trainY, test_size=0.1, random_state=0)


# Feature Matrix: Generate a linear vector for each 




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
    

    

