import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Model(object):
    def calculate_loss(self, X, y, reg_lambda, model):
        num_examples = X.shape[0] # row
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)
        # Add regulatization
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./num_examples * data_loss

    def predict(self, model, x):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def build_model(self, X, y, testX, testY, nn_input_dim, nn_output_dim, epsilon, reg_lambda, nn_hdim, num_passes=200, print_loss=False):
        # W3, b3 random
        # add in propagation
        # add in chain rule
        # tanh' = 1-tanh^2
        # y' = tanh(w2a1+b2)
        # y' = softmax(w2tanh(w1x+b1)+b2)

        num_examples = X.shape[0]
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))
     
        model = {}
        losses = []
        validation = []
        error = float('inf')
         
        # Gradient descent.
        for i in xrange(0, num_passes):
     
            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
     
            # Backpropagation
            delta3 = probs
            # y' - y
            delta3[range(num_examples), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)
     
            # Add regularization terms
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1
     
            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2
             
            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            
            # Verify if the testing error starts to increment, stop if it does
            if print_loss and i%10 == 0:
                newError = self.calculate_loss(testX, testY, reg_lambda, model)

                if newError > error:
                    print i, newError
                    break
                else:
                    error = newError

            if print_loss and i % 10 == 0:
                predict_train_y = self.predict(model, X)
                trainacc = accuracy_score(y, predict_train_y)
                losses.append(trainacc)
    
                predict_test_y = self.predict(model, testX)
                testacc = accuracy_score(testY, predict_test_y)
                validation.append(testacc)

                print "Loss after iteration %i: %f, %f" %(i, trainacc, testacc)

        
        plt.plot(losses)
        plt.plot(validation)
        plt.show()

        print W1, b1, W2, b2

        return model