# data.py
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# ================= Load MNIST =================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Shuffle dataset
np.random.seed()
# Shuffle training set
train_indices = np.arange(len(X_train))
np.random.shuffle(train_indices)
X_train = X_train[train_indices]
y_train = y_train[train_indices]

# Shuffle test set
test_indices = np.arange(len(X_test))
np.random.shuffle(test_indices)
X_test = X_test[test_indices]
y_test = y_test[test_indices]


# Optional: small sample for faster testing
X_train = X_train[:1000]
y_train = y_train[:1000]

X_test = X_test[:200]
y_test = y_test[:200]

# ================= Preprocessing =================
# Normalize to [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# ================= Plotting functions =================
def plot_sample(index=0):
    """Plot a single MNIST image"""
    plt.imshow(X_train[index].squeeze(), cmap='gray')  # squeeze removes channel dim
    plt.title(f"Label: {y_train[index]}")
    plt.axis('off')
    plt.show()


def plot_grid(n=10):
    """Plot a grid of MNIST images"""
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X_train[i].squeeze(), cmap='gray')
        plt.title(y_train[i])
        plt.axis('off')
    plt.show()


# ================= Optional: label conversion for binary classification =================
# Example: 0-4 -> 0, 5-9 -> 1
y_train_binary = (y_train >= 5).astype(int)
y_test_binary = (y_test >= 5).astype(int)
