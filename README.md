# Comp551_project3
McGill Applied Machine Learning(COMP551) Project 3

###################################################
cnn_keras_1.py:
This code creates a Convnet by using Keras, it uses data Augmentation and Dropout to mitigate overfitting.
- create a folder in you codes directory and name it files
- put all your .npy files inside that folder and run the code
- At the end code returns a .h5 file for model's weights that you can use for prediction on test set.
- you can then use cnn_prediction.py to use this weights for prediction on test set and save output as .csv file

cnn_prediction.py:
- befor using this you need weights as .h5 file from cnn_keras_1.py code.

logistic_regression_color_histogram.py:
- create a folder in you codes directory and name it files
- put all your .npy files inside that folder and run the code

nnModel.py includes the model used to construct the single-layered neural network

nn.py includes the code to run the model

The global variables are parameters that the user can change

Execute'python nn.py' to run the file
