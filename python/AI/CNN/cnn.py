from data import X_train, y_train, X_test, y_test, plot_sample, plot_grid
import numpy as np
from utility import (
    accuracy,
    precision_recall,
    relu, relu_diff,
    cce
)


# ================= Usage Examples =================

# single sample
plot_sample(0)

# 10 samples
plot_grid(10)

# direct data access
print(X_train.shape, y_train.shape)


class CNN:

    def __init__(self, input_shape, conv_layers=None, dense_layers=None, output_size=10, output_activation=None, loss_function=None):
        """
        CNN __init__ Arguments

        input_shape       : input image shape (H, W, C), e.g., '(28,28,1)'
        conv_layers       : list of conv layers (num_filters, kernel, stride, padding), e.g., '[(8,3,1,0)]'
        dense_layers      : list of dense layer sizes, e.g., '[128,64]'
        output_size       : number of classes, e.g., '10'
        output_activation : activation for output layer, e.g., 'softmax
        loss_function     : loss function to optimize, e.g., 'CCE'
        """

        self.input_shape = input_shape
        self.conv_layers = conv_layers or []
        self.dense_layers = dense_layers or []
        self.output_size = output_size
        self.output_activation = output_activation
        self.loss_function = loss_function

        # store weights and biases
        self.conv_weights = []      # convolutional layer weights
        self.conv_biases = []
        self.dense_weights = []     # dense layer weights
        self.dense_biases = []

        # history
        self.loss_train = []
        self.loss_val = []
        self.accuracies_train = []
        self.accuracies_val = []

        # initialize layers
        self._initialize_layers()

    def _initialize_layers(self):
        """
        Placeholder: initialize conv + dense layers based on configs.
        For now, just pass. Will implement weight initialization later.
        """

        # 1. Convolutional layers
        height, width, in_channels = self.input_shape
        for num_filters, kernel, stride, pad in self.conv_layers:
            # initialize weights and biases
            W = np.random.randn(num_filters, kernel, kernel, in_channels) * 0.01
            b = np.zeros(num_filters)

            self.conv_weights.append(W)
            self.conv_biases.append(b)

            # update spatial size after conv
            height = (height - kernel + 2 * pad) // stride + 1
            width = (width - kernel + 2 * pad) // stride + 1
            in_channels = num_filters  # output channels become input for next layer

        # flattened size after all conv layers
        flattened_size = height * width * in_channels

        # 2. Dense layers
        prev_units = flattened_size
        for layer_size in self.dense_layers:
            W = np.random.randn(prev_units, layer_size) * 0.01
            b = np.zeros(layer_size)

            self.dense_weights.append(W)
            self.dense_biases.append(b)

            prev_units = layer_size

        # 3. Output layer
        W = np.random.randn(prev_units, self.output_size) * 0.01
        b = np.zeros(self.output_size)
        self.dense_weights.append(W)
        self.dense_biases.append(b)


    def forward(self, X):
        """
        Placeholder: perform forward pass through conv layers -> dense layers -> output
        Should return final predictions and optionally intermediate activations.
        """
        pass

    def backward(self, X, y_true):
        """
        Placeholder: perform backpropagation to compute gradients and update weights
        """
        pass

    def fit(self, X_train, y_train, lr=0.01, epochs=10, batch_size=None,
            X_val=None, y_val=None, verbose=False):
        """
        Training loop: shuffling, batching, forward pass, backprop, metrics
        """
        self.lr = lr
        self.epochs_trained = 0

        for epoch in range(epochs):
            # TODO: shuffle X_train/y_train
            # TODO: split into batches
            # TODO: forward pass
            # TODO: backward pass / weight update
            # TODO: compute loss & accuracy for training and validation
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: ... - acc: ...")

    def predict(self, X):
        """
        Make predictions using the trained model
        """
        # TODO: forward pass
        return None

    def evaluate(self, X, y_true):
        """
        Compute accuracy / precision / recall
        """
        y_pred = self.predict(X)
        acc = accuracy(y_pred, y_true)
        prec, rec = precision_recall(y_pred, y_true)
        return acc, prec, rec