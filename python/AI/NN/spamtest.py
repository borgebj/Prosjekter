from neural_network import NeuralNet
from utility import relu_act, sigmoid_act, bce_loss
from utility import get_predictions, print_prediction
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
        free = np.random.randint(0, 4)      # maybe 0-3 occurrences
        win = np.random.randint(0, 2)       # maybe 0-1 occurrences
        offer = np.random.randint(0, 3)     # maybe 0-2 occurrences

        X_aug.append([free, win, offer])

        # rule:  if at least 2 keywords present, mark as spam
        y_aug.append([1 if sum([free>0, win>0, offer>0]) >= 2 else 0])

    return np.array(X_aug), np.array(y_aug)


# ------------------ Data ------------------
np.set_printoptions(precision=4, suppress=True)

# Original data
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

X_extra, y_extra = generate_samples(20)    # 20 synthetic samples
X = np.vstack([X, X_extra])
y = np.vstack([y, y_extra])

# shuffle data
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

# split sizes
train_split = int(0.6 * len(X))  # 60% training         (seen data)
dev_split = int(0.2 * len(X))    # 20% validation       (hyperparameter tuning etc)
test_split = dev_split           # 20% testing          (unseen data)

# data split
X_train = X_shuffled[:train_split]
y_train = y_shuffled[:train_split]

X_dev = X_shuffled[train_split:train_split+dev_split]
y_dev = y_shuffled[train_split:train_split+dev_split]

X_test = X_shuffled[train_split+dev_split:]
y_test = y_shuffled[train_split+dev_split:]

# ------------------ Neural Network ------------------
nn = NeuralNet(
    input_size=3, hidden_size=[2], output_size=1,       # 3-2-1 architecture
    hidden_activation=relu_act,
    output_activation=sigmoid_act,                      # relu + sigmoid + bce
    loss_function=bce_loss,
    lr=0.1                                              # 0.1 learning rate
)


# ------------------ Training ------------------
nn.train(X_train, y_train, epochs=500)   # raw data, no manual normalization

# saves to file
nn.save("spam_detector.pt")


# ------------------ Print Results ------------------
probs_train, classes_train = get_predictions(nn, X_train)
print(f"\n\nTrain data ({train_split})")
print_prediction(X_train, y_train, probs_train, classes_train)

probs_dev, classes_dev = get_predictions(nn, X_dev)
print(f"\nDev data ({dev_split})")
print_prediction(X_dev, y_dev, probs_dev, classes_dev)

probs_test, classes_test = get_predictions(nn, X_test)
print(f"\nTest data ({test_split})")
print_prediction(X_test, y_test, probs_test, classes_test)