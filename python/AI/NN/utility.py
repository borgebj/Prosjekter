import numpy as np

"""

Verktøy for bruk av neural network.
Inneholder aktiveringsfunksjoner, tapsfunksjoner, normalisering og one-hot encoding.

For mer info, sjekk OneNote-dokumentet "IN4050/Forelesninger/Neural Networks and Back Propagation".

"""


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
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(x):
    """Not used due to CCE loss"""
    s = softmax(x)
    return s * (1 - s)


# ------------------------------------------------------


# ========== Loss function ==================
# ------------------------------------------------------
def binary_cross_entropy(y_true, y_pred):
    """Calculates loss with binary cross entropy"""
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def derivative_bce(y_true, y_pred):
    """Calculates derivative of binary cross entropy loss"""
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


# ------------------------------------------------------


def categorical_cross_entropy(y_true, y_pred):
    """Calculates loss with categorical cross entropy"""
    cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return cce


def derivative_cce(y_true, y_pred):
    """Calculates derivative of categorical cross entropy loss"""
    N = y_true.shape[0]
    return - (y_true / y_pred) / N


# ------------------------------------------------------


# ========== Data Normalization ==================
# ------------------------------------------------------
def normalize(x, x_min, x_max):
    """Min-Max Normalization (0-1) with numpy array x"""
    return (x - x_min) / (x_max - x_min)


# ------------------------------------------------------


# ========== Label Preparation ==================
# ------------------------------------------------------
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


# ------------------------------------------------------

import numpy as np

"""

Verktøy for bruk av neural network.
Inneholder aktiveringsfunksjoner, tapsfunksjoner, normalisering og one-hot encoding.

For mer info, sjekk OneNote-dokumentet "IN4050/Forelesninger/Neural Networks and Back Propagation".

"""


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
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(x):
    """Not used due to CCE loss"""
    s = softmax(x)
    return s * (1 - s)


# ------------------------------------------------------


# ========== Loss function ==================
# ------------------------------------------------------
def binary_cross_entropy(y_true, y_pred):
    """Calculates loss with binary cross entropy"""
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def derivative_bce(y_true, y_pred):
    """Calculates derivative of binary cross entropy loss"""
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


# ------------------------------------------------------


def categorical_cross_entropy(y_true, y_pred):
    """Calculates loss with categorical cross entropy"""
    cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return cce


def derivative_cce(y_true, y_pred):
    """Calculates derivative of categorical cross entropy loss"""
    N = y_true.shape[0]
    return - (y_true / y_pred) / N


# ------------------------------------------------------


# ========== Data Normalization ==================
# ------------------------------------------------------
def normalize(x, x_min, x_max):
    """Min-Max Normalization (0-1) with numpy array x"""
    return (x - x_min) / (x_max - x_min)


# ------------------------------------------------------


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