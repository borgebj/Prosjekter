import numpy as np

"""

Verktøy for bruk av neural network.
Inneholder aktiveringsfunksjoner, tapsfunksjoner, normalisering og one-hot encoding.

For mer info, sjekk OneNote-dokumentet "IN4050/Forelesninger/Neural Networks and Back Propagation".

"""


# ========= Printing functions ==================
# ------------------------------------------------------
def get_predictions(nn, X, threshold=0.5):
    """Return predicted probabilities and classes from a neural network."""
    probs = nn.predict(X)
    classes = ["Spam" if p >= threshold else "Not Spam" for p in probs.flatten()]
    return probs, classes


# ------------------------------------------------------
def print_prediction(X, y_true, probs, classes):
    """Print predictions alongside true labels and feature values."""
    print("\n========== Testing Predictions ==========")
    print("Data\t  True\t  Prediction")
    for features, label, prob, cls in zip(X, y_true, probs, classes):
        print(f"{features} = [{label[0]}] --> Pred: {prob[0]*100:6.2f}% ({cls})")


# ------------------------------------------------------


# ========= Activation functions ==================
# ------------------------------------------------------
def relu(x):
    """ReLU activation function in hidden layer"""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU used in backpropagation"""
    return np.where(x > 0, 1, 0)


# ------------------------------------------------------


def sigmoid(x):
    """Sigmoid activation function in output layer"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Not used due to BCE loss"""
    fx = sigmoid(x)
    return fx * (1 - fx)


# ------------------------------------------------------


def softmax(x):
    """Softmax activation function for multi-class output"""
    # axis=1 ensures function applied across each row
    # keepdims ensures original dimensions maintained
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def softmax_derivative(x):
    """Not used due to CCE loss"""
    s = softmax(x)
    return s * (1 - s)


# ------------------------------------------------------


# ========== Loss function ==================
# ------------------------------------------------------
def mse(y_true, y_pred):
    """MSE loss to present losses across epochs
    Step 1-2 includes loss for single sample"""
    # 1. calculates error (y - p)
    # 2. squares errors ^2
    # 3. sums errors        (numpy internal)
    # 4. averages           (numpy internal)
    return np.mean((y_true - y_pred) ** 2)


def derivative_mse(y_true, y_pred):
    """Calculates derivative of MSE"""
    N = y_true.shape[0]
    return 2 * (y_pred - y_true) / N

# ------------------------------------------------------


def binary_cross_entropy(y_true, y_pred):
    """Calculates loss with binary cross entropy"""
    # (1/N) * (y * log(p) + (1-y) * log(1-p))
    y_pred += 1e-15
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def derivative_bce(y_true, y_pred):
    """Calculates derivative of binary cross entropy"""
    # (1/N) * (p - y) / p * (1- p)
    y_pred += 1e-15
    N = y_true.shape[0]
    return (1/N) * (y_pred - y_true) / (y_pred * (1 - y_pred))


# ------------------------------------------------------


def categorical_cross_entropy(y_true, y_pred):
    """Calculates loss with categorical cross entropy"""
    # (1/N) * sum(y * log(p))       (sums over classes)
    eps = 1e-15
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


def derivative_cce(y_true, y_pred):
    """Calculates derivative of categorical cross entropy"""
    # (1/N) * (y - p)
    y_pred += 1e-15
    N = y_true.shape[0]
    return - (y_true / y_pred) / N


# ------------------------------------------------------


# ========== Data Normalization ==================
# ------------------------------------------------------
def minmax(x, x_min, x_max):
    """Min-Max Normalization (0-1) with numpy array x"""
    eps = 1e-8  # small val to avoid division by zero
    return (x - x_min) / (x_max - x_min + eps)


def standard(X, x_min, x_max):
    """Standard scaler aka Z-score"""
    # do axis=0 > column, due to per-feature
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


# ========== Label Preparation ==================
# ------------------------------------------------------
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


# ------------------------------------------------------


# ==================== Containers ====================
class Activation:
    """Container for activation + derivative"""

    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative


class Loss:
    """Container for loss + derivative"""

    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative


# ==================== Activation Instances ====================
relu_act = Activation(relu, relu_derivative)
sigmoid_act = Activation(sigmoid, sigmoid_derivative)
softmax_act = Activation(softmax, softmax_derivative)

# ==================== Activation Instances ====================
bce_loss = Loss(binary_cross_entropy, derivative_bce)
cce_loss = Loss(categorical_cross_entropy, derivative_cce)