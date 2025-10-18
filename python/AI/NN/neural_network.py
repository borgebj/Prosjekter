from utility import normalize, one_hot
from utility import relu_act, sigmoid_act, softmax_act
from utility import bce_loss, cce_loss
from utility import *

import numpy as np


def print_prediction(nn, test, true, train_min, train_max):
    print("\n========== Testing Predictions ==========")
    norm = normalize(test, train_min, train_max)    # norm
    prediction = nn.predict(norm)                   # predict

    for i in range(test.shape[0]):
        features = test[i]
        label = true[i, 0]

        pred_prob = prediction[i, 0]
        pred_class = "Spam" if pred_prob >= 0.5 else "Not Spam"

        # Combine features and label in one line
        print(f"{features} = [{label}]  -->  Pred: {pred_prob*100:.2f}% ({pred_class})")


class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size,
                 hidden_activation=relu_act,
                 output_activation=sigmoid_act,
                 loss_function=bce_loss,
                 lr=0.01):

        self.weights = []
        self.biases = []

        layer_size = [input_size] + hidden_size + [output_size]

        # initialize weights and biases (based on all layers)
        for i in range(len(layer_size) - 1):
            self.weights.append(np.random.randn(layer_size[i], layer_size[i+1]) * 0.01)
            self.biases.append(np.zeros(layer_size[i+1]))

        self.lr = lr                                # learning rate
        self.hidden_activation = hidden_activation  # hidden layer activation function
        self.output_activation = output_activation  # output layer activation function
        self.loss_function = loss_function          # loss function

        # input layer to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)

        # hidden layer to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        """Forward pass input --> output"""
        activations = [X]        # store A
        pre_activations = []     # store Z

        for i in range(len(self.weights)):

            # Z = XW + b
            Z = activations[-1] @ self.weights[i] + self.biases[i]
            # [-1] gets previous layers
            pre_activations.append(Z)

            if i == len(self.weights) - 1: # output layer
                A = self.output_activation.func(Z)
            else:
                A = self.hidden_activation.func(Z)

            activations.append(A)

        return pre_activations, activations

    def backward(self, X, y_true, Z1, Z2, A1, A2):
        """Updating weights and bias by going backward input <-- Output"""

        N = y_true.shape[0]  # number of samples

        # derivatives
        loss_deriv = self.loss_function.derivative
        output_deriv = self.output_activation.derivative
        hidden_deriv = self.hidden_activation.derivative

        # output layer
        dA2 = loss_deriv(y_true, A2)     # L/A
        dZ2 = dA2 * output_deriv(Z2)     # L/Z
        dW2 = A1.T @ dZ2 / N             # L/W
        dB2 = np.sum(dZ2, axis=0) / N    # L/b

        # hidden layer
        dA2 = dZ2 @ self.W2.T            # L/A
        dZ1 = dA2 * hidden_deriv(Z1)     # L/Z
        dW1 = X.T @ dZ1 / N              # L/W
        dB1 = np.sum(dZ1, axis=0) / N    # L/b

        # output update
        self.W2 -= self.lr * dW2         # weight update
        self.b2 -= self.lr * dB2         # bias update

        # hidden update
        self.W1 -= self.lr * dW1         # weight update
        self.b1 -= self.lr * dB1         # bias update

    def train(self, X, Y, epochs=100):
        """Train by doing forward and backward passes"""

        for epoch in range(epochs):             # training epochs
            Z1, Z2, A1, A2 = self.forward(X)        # forward pass
            self.backward(X, Y, Z1, Z2, A1, A2)     # backward pass

            # training loop
            if (epoch + 1) % 50 == 0 or epoch == 0:
                loss = binary_cross_entropy(Y, A2)
                print(f"Epoch {epoch + 1:3}, Loss: {loss:.4f}")

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return A2


"""
Usage for loading the model  (name "spam_model.npz")

data = np.load("spam_model.npz")
nn.W1 = data["W1"]
nn.b1 = data["b1"]
nn.W2 = data["W2"]
nn.b2 = data["b2"]
train_min = data["train_min"]
train_max = data["train_max"]
"""