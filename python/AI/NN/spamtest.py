from utility import get_predictions, print_prediction
from utility import relu_act, leaky_relu_act, sigmoid_act, bce_loss
from neural_network import NeuralNet
import numpy as np


"""
3-2-1 NN for spam detection using features "free", "win", "offer":

  Input Layer     Hidden Layer   Output Layer
  (3 neurons)     (2 neurons)    (1 neuron)

   [X1] ──────┐     [H1]───┐
   [X2] ──────┼─►────┤     ├──►──[O1]
   [X3] ──────┘     [H2]───┘
"""


def generate_samples(n):
    X_aug = []
    y_aug = []

    for _ in range(n):
        free = np.random.randint(0, 6)
        win = np.random.randint(0, 4)
        offer = np.random.randint(0, 4)

        X_aug.append([free, win, offer])

        score = 0.6 * free + 1.0 * win + 0.7 * offer

        if free > 0 and win > 0:
            score += 1.0

        if win > 0 and offer > 0:
            score += 0.5

        score += np.random.normal(0, 0.5)

        # spam threshold
        is_spam = score >= 3.5
        y_aug.append([int(is_spam)])

    return np.array(X_aug), np.array(y_aug)



# ------------------ Data ------------------
np.set_printoptions(precision=4, suppress=True)

# Original data
# 12 samples, 3 features (keywords)
# first feature = "free",
# second feature = "win",
# third feature = "offer"
X = np.array([
    [2, 0, 1],  # "free" and "offer"
    [0, 1, 0],  # "win"
    [1, 0, 0],  # "free"
    [3, 1, 2],  # "free", "win", and "offer"
    [0, 0, 0],  # no keywords
    [1, 1, 1],  # all keywords -> likely spam
    [0, 1, 1],  # "win" and "offer"
    [1, 0, 1],  # "free" and "offer"
    [0, 0, 0],  # no keywords
    [0, 1, 0],  # "win" only
    [1, 0, 0],  # "free" only
    [0, 0, 1],  # "offer" only
])

# Labels: spam = 1, not spam = 0
y = np.array([
    [1],  # spam
    [1],  # spam
    [0],  # not spam
    [1],  # spam
    [0],  # not spam
    [1],  # spam
    [1],  # spam
    [1],  # spam
    [0],  # not spam
    [1],  # spam
    [0],  # not spam
    [0],  # not spam
])

# generate 500 additional samples for training
X_extra, y_extra = generate_samples(500)

# combine original and generated data
X = np.vstack([X, X_extra])
y = np.vstack([y, y_extra])

# shuffle data
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

# split boundaries
# 60% training, 20% validation, 20% testing
train_end = int(0.6 * len(X))
dev_end = train_end + int(0.2 * len(X))

# training data split (seen) = 60%
X_train = X_shuffled[:train_end]
y_train = y_shuffled[:train_end]

# validation data split (unseen) = 20%
X_dev = X_shuffled[train_end:dev_end]
y_dev = y_shuffled[train_end:dev_end]

# test data split (unseen) = 20%
X_test = X_shuffled[dev_end:]
y_test = y_shuffled[dev_end:]

# Z-score feature normalization
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8

X_train_scaled = (X_train - X_mean) / X_std
X_dev_scaled = (X_dev - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# ------------------ Neural Network ------------------
nn = NeuralNet(
    input_size=3, hidden_size=[4], output_size=1,       # 3-4-1 architecture
    hidden_activation=leaky_relu_act,
    output_activation=sigmoid_act,                      # leaky_relu + sigmoid + bce
    loss_function=bce_loss,
    lr=0.01                                             # 0.01 learning rate
)

# ------------------ Training ------------------
nn.train(X_train_scaled, y_train, epochs=10000)  # train on normalized training data

# saves to file
nn.save("spam_detector.pt")                                # save model weights and biases
np.savez("scaler_params.npz", mean=X_mean, std=X_std)  # save normalization params


# ------------------ Print Results ------------------
pred_train, classes_train = get_predictions(nn, X_train_scaled)
print(f"\n\nTrain data (seen) ({len(X_train)})")
print_prediction(X_train, y_train, pred_train, classes_train)

pred_dev, classes_dev = get_predictions(nn, X_dev_scaled)
print(f"\nDev data (unseen) ({len(X_dev)})")
print_prediction(X_dev, y_dev, pred_dev, classes_dev)

pred_test, classes_test = get_predictions(nn, X_test_scaled)
print(f"\nTest data (unseen) ({len(X_test)})")
print_prediction(X_test, y_test, pred_test, classes_test)

nn.evaluate(X_test_scaled, y_test)