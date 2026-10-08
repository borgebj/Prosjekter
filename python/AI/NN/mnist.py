import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # Linear layer = calculates the pre-activation
        # z = xW^T + b

        # Bias is automatically included in nn.Linear

        # 784 input -> 128 neurons
        self.layer1 = nn.Linear(784, 128)

        # 128 neurons -> 64 neurons
        self.layer2 = nn.Linear(128, 64)

        # 64 neurons -> 10 outputs (digits 0-9)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        """Does forward propagation with input X"""

        # flattens input:  [1, 28, 28] -> [784]
        x = self.flatten(x)

        # ==== HIDDEN LAYER 1 ====

        # x becomes z (pre-activation)
        z1 = self.layer1(x)

        # z becomes a (activation)
        a1 = torch.relu(z1)

        # ==== LAYER 2 ====
        z2 = self.layer2(a1)
        a2 = torch.relu(z2)

        # ==== OUTPUT LAYER 3 ====
        out = self.layer3(a2)

        return out



# ========== DATA ==========

# converts MNISt image to PyTorch tensor
# image has shape [1, 28, 28]
#   1 = grayscale channel
#   28 = height
#   28 = width
transform = transforms.ToTensor()

# MNIST - train and test data
train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# dataloaders with batches
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)


# ========== MODEL SETUP==========

# Model
model = MNISTModel()

# loss function: BCE
loss_fn = nn.CrossEntropyLoss()

# optimizer: updates weights
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ========== TRAINING ==========
epochs = 5

for epoch in range(epochs):

    # put model in training mode
    model.train()

    for images, labels in train_loader:

        # forward pass
        logits = model(images)

        # measure loss using BCE
        loss = loss_fn(logits, labels)

        # clear gradients from previous iteration
        optimizer.zero_grad()

        # calculate new gradients
        loss.backward()

        # update weights, essentially handling:
        #   w -= lr * dW
        #   b -= lr * dB
        optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {loss.item():.4f}"
    )

# put model in evaluation mode
model.eval()

correct = 0
total = 0

# goes through all samples, predicts and checks
with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images)
        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Accuracy: {accuracy:.2f}%")