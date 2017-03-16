from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
import nnModel
from sklearn.metrics import accuracy_score

NN_INPUT_DIM = 64*64*3
NN_OUTPUT_DIM = 40
# learning rate
EPSILON = 0.000005
# regulization
REG_LAMBDA = 0.01
# number of hidden nodes
NN_HIDDEN_NUM = 5000

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

train_X, test_X, train_Y, test_Y = train_test_split(X, raw_Y)

train_X = train_X[0:5000][:][:][:]
train_Y = train_Y[0:5000]
test_X = test_X[5000:6000][:][:][:]
test_Y = test_Y[5000:6000]

# Fit model
nn = nnModel.Model()
model = nn.build_model(train_X, train_Y, test_X, test_Y, NN_INPUT_DIM, NN_OUTPUT_DIM, EPSILON, REG_LAMBDA, NN_HIDDEN_NUM, print_loss=True)

# Predict training data
predict_Y = nn.predict(model, test_X)

print "accuracy"
print accuracy_score(test_Y, predict_Y)










