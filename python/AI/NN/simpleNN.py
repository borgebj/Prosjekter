import numpy as np


class NeuralNet:
    """
    Simple feed-forward neural network.

    ReLU hidden layers
    Sigmoid output layer
    Mean Squared Error loss
    Gradient descent
    """

    def __init__(self, input_size, hidden_sizes, output_size, lr=0.01):
        self.lr = lr

        # Layer sizes: input -> hidden -> output
        sizes = [input_size] + hidden_sizes + [output_size]

        # Initialize weights and biases
        self.weights = [
            np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2 / sizes[i])
            for i in range(len(sizes) - 1)
        ]

        # Initialize biases to zeros
        self.biases = [
            np.zeros(size)
            for size in sizes[1:]
        ]

    # ---------- Activation functions ----------

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    # ---------- Loss function ----------
    def mse(self, X, Y):
        return np.mean((X - Y) ** 2)

    def mse_derivative(self, X, Y):
        return 2 * (X - Y) / len(Y)

    # ---------- Forward pass ----------

    def forward(self, X):

        # list of activations and pre-activations for each layer
        activations = [
            X,          # input layer
        #   A1          # hidden layer 1
        #   A2          # hidden layer 2
        #   A3          # output layer
        ]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):

            # z = X * W + b
            # applies linear transformation to the input of the layer
            Z = activations[-1] @ W + b
            pre_activations.append(Z)

            # Sigmoid for output layer - value between 0 and 1
            if i == len(self.weights) - 1:
                A = self.sigmoid(Z)

            # ReLU for hidden layers - value between 0 and infinity
            else:
                A = self.relu(Z)

            activations.append(A)

        return activations, pre_activations

    # ---------- Backpropagation ----------

    def backward(self, y, activations, pre_activations):
        N = len(y)

        # --------------------------------------------------
        # Start at the output layer
        # --------------------------------------------------

        prediction = activations[-1]

        # derivative of loss function w.r.t. predictions (MSE)
        dA = self.mse_derivative(prediction, y)

        for i in reversed(range(len(self.weights))):

            # Activation derivative
            if i == len(self.weights) - 1:
                dZ = dA * self.sigmoid_derivative(pre_activations[i])
            else:
                dZ = dA * self.relu_derivative(pre_activations[i])

            # Gradients
            dW = activations[i].T @ dZ
            db = np.sum(dZ, axis=0)

            # Update parameters
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

            # Gradient for previous layer
            dA = dZ @ self.weights[i].T

    # ---------- Training ----------

    def train(self, X, y, epochs=1000):

        # Train the neural network using gradient descent
        for epoch in range(epochs):

            # Forward
            # produces the predictions
            activations, pre_activations = self.forward(X)

            # Loss
            # measures how wrong predictions are
            prediction = activations[-1]
            loss = self.mse(prediction, y)

            # Backward
            # updates the weights and biases based on the loss
            self.backward(y, activations, pre_activations)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # ---------- Prediction ----------

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]


# ---------- Test ----------

if __name__ == "__main__":

    network = NeuralNet(
        input_size=2,
        hidden_sizes=[3],
        output_size=1
    )

    # 2 features, 3 samples, 1 output
    X = np.array([
        [2.0, 3.0],
        [1.0, 4.0],
        [5.0, 2.0]
    ])

    targets = np.array([
        [1.0],
        [0.0],
        [1.0]
    ])

    print("\nBefore training:")
    print(network.predict(X), end="\n\n")

    network.train(X, targets, epochs=1000)

    print("\nAfter training:")
    print(network.predict(X))

    print("\nGround truths:")
    print(targets)