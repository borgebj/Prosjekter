from utility import relu_act, sigmoid_act, softmax_act
from utility import bce_loss, cce_loss
import numpy as np
import pickle


class NeuralNet:
    def __init__(self,
                 input_size, hidden_size, output_size,
                 hidden_activation, output_activation, loss_function,
                 lr=0.01):

        self.lr = lr                                # learning rate
        self.hidden_activation = hidden_activation  # hidden layer activation function
        self.output_activation = output_activation  # output layer activation function
        self.loss_function = loss_function          # loss function

        # weights and biases storage
        self.weights = []
        self.biases = []

        # storage for normalization - set on first training
        self.x_min = None
        self.x_max = None

        # if number provided, make list
        if isinstance(hidden_size, int): hidden_size = [hidden_size]
        layer_size = [input_size] + hidden_size + [output_size]

        # initialize weights and biases (based on all layers) (based on activation)
        for i in range(len(layer_size) - 1):
            if hidden_activation == relu_act:
                scale = np.sqrt(2 / layer_size[i])  # He initialization
            else:
                scale = np.sqrt(1 / layer_size[i])  # Xavier initialization

            self.weights.append(np.random.randn(layer_size[i], layer_size[i + 1]) * scale)
            self.biases.append(np.zeros(layer_size[i + 1]))

    def normalize(self, x):
        """Min-Max Normalization (0-1) with numpy array x"""
        eps = 1e-8  # small val to avoid division by zero
        return (x - self.x_min) / (self.x_max - self.x_min + eps)

    def forward(self, X):
        """Forward pass input --> output"""
        activations = [X]        # store A
        pre_activations = []     # store Z

        # for each layer (hidden + output)
        for i in range(len(self.weights)):

            # Z = XW + b
            Z = activations[-1] @ self.weights[i] + self.biases[i]
            # [-1] gets previous layers
            pre_activations.append(Z)

            if i == len(self.weights) - 1: # output layer
                A = self.output_activation.func(Z)
            else:
                A = self.hidden_activation.func(Z)

            # add A to list
            activations.append(A)

        return pre_activations, activations


    def backward(self, y_true, pre_acts, acts):
        N = y_true.shape[0]  # number of samples
        dA = self.loss_function.derivative(y_true, acts[-1])  # L/A   (start at output A)

        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dZ = dA * self.output_activation.derivative(pre_acts[i])  # L/Z - output
            else:
                dZ = dA * self.hidden_activation.derivative(pre_acts[i])  # L/Z - hidden

            dW = acts[i].T @ dZ / N          # L/W
            dB = np.sum(dZ, axis=0) / N      # L/b

            # update weights and biases
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * dB

            dA = dZ @ self.weights[i].T        # L/A for next layers (output -> hidden -> input)


    def train(self, X, Y, epochs=100):
        """
        Train by doing forward and backward passes
        Expects raw data - does internal normalization
        """
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)

        # normalize input
        X_norm = self.normalize(X)

        # training epochs
        for epoch in range(epochs):
            pre_acts, acts = self.forward(X_norm)            # forward pass
            self.backward(Y, pre_acts, acts)                 # backward pass

            if (epoch + 1) % 50 == 0 or epoch == 0:
                loss = self.loss_function.func(Y, acts[-1])  # use last A as output
                print(f"Epoch {epoch+1:3} - Loss: {loss:.4f}")


    def predict(self, X):
        X_norm = self.normalize(X)
        _, activations = self.forward(X_norm)
        return activations[-1]  # return output layer activations

    def save(self, filename="model.pt"):
        """Saves model parameters and config to file"""
        model_data = {
            "lr": self.lr,
            "weights": self.weights,
            "biases": self.biases,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "hidden_activation": self.hidden_activation.func.__name__,
            "output_activation": self.output_activation.func.__name__,
            "loss_function": self.loss_function.func.__name__,
        }
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to {filename}")

    @staticmethod
    def load(filename="model.pt"):
        """Loads model parameters and config from file"""
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        # map names back to function-objects
        act_map = {"relu": relu_act, "sigmoid": sigmoid_act, "softmax": softmax_act}
        loss_map = {"binary_cross_entropy": bce_loss, "categorical_cross_entropy": cce_loss}

        # get layer sizes from weights
        input_size = model_data["weights"][0].shape[0]
        hidden_sizes = [w.shape[1] for w in model_data["weights"][:-1]]
        output_size = model_data["weights"][-1].shape[1]

        # new, loaded NN
        nn = NeuralNet(
            input_size=input_size,
            hidden_size=hidden_sizes,
            output_size=output_size,
            hidden_activation=act_map[model_data["hidden_activation"]],
            output_activation=act_map[model_data["output_activation"]],
            loss_function=loss_map[model_data["loss_function"]],
            lr=model_data["lr"]
        )

        # restore weights, biases, normalization
        nn.weights = model_data["weights"]
        nn.biases = model_data["biases"]
        nn.x_min = model_data["x_min"]
        nn.x_max = model_data["x_max"]

        print(f"✅ Model loaded from {filename}")
        return nn
