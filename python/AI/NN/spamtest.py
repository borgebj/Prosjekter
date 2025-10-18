from neural_network import NeuralNet, normalize, print_prediction
import numpy as np

"""
3-2-1 NN for spam detection using features "free", "win", "offer":

  Input Layer     Hidden Layer   Output Layer
  (3 neurons)     (2 neurons)    (1 neuron)

   [X1] ──────┐     [H1]───┐
   [X2] ──────┼─►────┤     ├──►──[O1]
   [X3] ──────┘     [H2]───┘
"""

np.set_printoptions(precision=4, suppress=True)

# Input
X_orig = np.array([
    [2, 0, 1],  # "free" and "offer"
    [0, 1, 0],  # "win"
    [1, 0, 0],  # "free"
    [3, 1, 2],  # "free", "win", and "offer"
    [0, 0, 0],  # no keywords
])

# Normalize
train_min = X_orig.min(axis=0)  # column-wise min  (min for training data)
train_max = X_orig.max(axis=0)  # column-wise max  (max for training data)
X = normalize(X_orig, train_min, train_max)

# Labels: spam = 1, not spam = 0
y_true = np.array([
    [1],  # spam
    [1],  # spam
    [0],  # not spam
    [1],  # spam
    [0],  # not spam
])

# neural net for spam detection
nn = NeuralNet(input_size=3, hidden_size=2, output_size=1, lr=0.1)
nn.train(X, y_true, epochs=500)
np.savez("spam_model.npz", W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2,  # save model
         train_min=train_min, train_max=train_max)

# New test samples (unseen during training)
X_test = np.array([
    [1, 1, 1],  # all keywords present -> likely spam
    [0, 1, 1],  # "win" and "offer" -> probably spam
    [1, 0, 1],  # "free" and "offer" -> likely spam
    [0, 0, 0],  # no keywords -> not spam
    [0, 1, 0],  # "win" only -> spam? borderline
    [1, 0, 0],  # "free" only -> borderline
    [0, 0, 1],  # "offer" only -> borderline
])
y_true_test = np.array([
    [1],  # spam
    [1],  # spam
    [1],  # spam
    [0],  # not spam
    [1],  # spam
    [0],  # not spam
    [0],  # not spam
])

# (normalization done inside)
print("\nSeen data")
print_prediction(nn, X_orig, y_true, train_min, train_max)        # training data predictions
print("\nUnseen data")
print_prediction(nn, X_test, y_true_test, train_min, train_max)   # test data predictions