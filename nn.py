# from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
# from keras.models import Sequential
# from keras.layers import Activation
# from keras.optimizers import SGD
# from keras.layers import Dense
# from keras.utils import np_utils
# from imutils import paths
import numpy as np
import cv2
import nnModel

NN_INPUT_DIM = 64*64*3
NN_OUTPUT_DIM = 40
# learning rate
EPSILON = 0.01
# regulization
REG_LAMBDA = 0.01

def image_to_feature_vector(image, size=(64, 64)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

# Load data
raw_X = np.load("tinyX.npy")
raw_Y = np.load("tinyY.npy")

raw_X = np.reshape(raw_X, (len(raw_X), 64, 64, 3))

X = []

for img in raw_X:
    X.append(image_to_feature_vector(img))

# Normalize data
X = np.array(X)/255.0

train_X, test_X, train_Y, test_Y = train_test_split(X, raw_Y, random_state=42)

train_X = train_X[0:5000][:][:][:]
train_Y = train_Y[0:5000]
test_X = test_X[0:1000][:][:][:]
test_Y = test_Y[0:1000]

# Fit model
nn = nnModel.Model()
model = nn.build_model(train_X, train_Y, NN_INPUT_DIM, NN_OUTPUT_DIM, EPSILON, REG_LAMBDA, 3, print_loss=True)

# Predict training data
predict_Y = []

for x in test_X:
    nn.predict(model, x)

print float(np.sum(predict_Y, test_Y))/float(test_Y.shape[0])










