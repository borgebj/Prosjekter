from pathlib import Path
import numpy as np
from typing import Optional


# =====================================
# Weight Initialization

def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Initialize weights using Xavier normal initialization."""

    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_out, fan_in) * std


# =====================================
# Activation: Softmax

def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert logits into a numerically stable probability distribution."""

    scores = np.asarray(scores, dtype=np.float64)
    exp_scores = np.exp(scores - np.max(scores))

    return exp_scores / np.sum(exp_scores)


# =====================================
# Recurrent Neural Network

class RNN:

    def __init__(
            self,
            input_size: int = 2,
            hidden_size: int = 2,
            output_size: int = 5,
            lr: float = 0.01,
            max_grad_norm: Optional[float] = 5.0,
    ):
        """Initialize a many-to-one RNN."""

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lr = lr
        self.max_grad_norm = max_grad_norm

        # Input-to-hidden and hidden-to-hidden weights.
        self.Wxh = xavier_init(input_size, hidden_size)
        self.Whh = xavier_init(hidden_size, hidden_size)
        self.bh = np.zeros(hidden_size)

        # Hidden-to-output weights and output bias.
        self.Why = xavier_init(hidden_size, output_size)
        self.by = np.zeros(output_size)

        # Cache values from the most recent forward pass for BPTT.
        self.x_sequence = None
        self.hidden_states = None
        self.output_scores = None
        self.probabilities = None

    def forward(self, x_sequence: np.ndarray) -> np.ndarray:
        """Process a sequence and return output logits."""

        x_sequence = np.asarray(x_sequence, dtype=np.float64)
        self.x_sequence = x_sequence.copy()

        # initial hidden state (0)
        h = np.zeros(self.hidden_size)

        hidden_states = []

        # Process each input vector in order.
        for x in self.x_sequence:
            z = self.Wxh @ x + self.Whh @ h + self.bh
            h = np.tanh(z)
            hidden_states.append(h.copy())

        self.hidden_states = np.asarray(hidden_states)

        # logits from final hidden state
        final_hidden_state = self.hidden_states[-1]

        # pre-activation (z)
        self.output_scores = (self.Why @ final_hidden_state) + self.by

        # activation (a)
        self.probabilities = softmax(self.output_scores)

        return self.output_scores.copy()

    def backward(self, target_id: int) -> None:
        """Calculate gradients using BPTT and update model parameters."""

        # Output layer gradients: softmax + cross-entropy.
        d_scores = self.probabilities.copy()
        d_scores[target_id] -= 1.0

        # 2. calculates gradient for output-layer
        final_hidden_state = self.hidden_states[-1]
        dWy = np.outer(d_scores, final_hidden_state)
        dby = d_scores.copy()

        # Propagate output error into the final hidden state.
        dh = self.Why.T @ d_scores

        dWx = np.zeros_like(self.Wxh)
        dWh = np.zeros_like(self.Whh)
        db = np.zeros_like(self.bh)

        h_initial = np.zeros(self.hidden_size)

        # Backpropagation Through Time (BPTT).
        for t in reversed(range(len(self.x_sequence))):
            x = self.x_sequence[t]
            h = self.hidden_states[t]

            if t == 0:
                h_previous = h_initial
            else:
                h_previous = self.hidden_states[t - 1]

            # Derivative of tanh.
            dz = dh * (1.0 - h ** 2)

            # gradient at this time-step
            dWx += np.outer(dz, x)
            dWh += np.outer(dz, h_previous)
            db += dz

            # Propagate error to the previous hidden state.
            dh = self.Whh.T @ dz

        # Gradient clipping limits the size of large updates.
        gradients = [dWx, dWh, db, dWy, dby]

        # calculate size of all gradients
        grad_norm = np.sqrt(sum(np.sum(gradient ** 2) for gradient in gradients))

        # limits gradient size based on scale and largest size
        if self.max_grad_norm is not None and grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (grad_norm + 1e-12)
            gradients = [gradient * scale for gradient in gradients]

        dWx, dWh, db, dWy, dby = gradients

        # SGD (Stochastic Gradient Descent):
        # - update weights using gradients from this training example to reduce loss.
        self.Wxh -= self.lr * dWx
        self.Whh -= self.lr * dWh
        self.bh -= self.lr * db

        self.Why -= self.lr * dWy
        self.by -= self.lr * dby

    def predict(self, x_sequence: np.ndarray) -> tuple[int, np.ndarray]:
        """Return the predicted class ID and its probability distribution."""

        self.forward(x_sequence)

        # highest probability as int-id
        prediction_id = int(np.argmax(self.probabilities))

        return prediction_id, self.probabilities.copy()

    def compute_loss(self, target_id: int) -> float:
        """Calculate cross-entropy loss for the target class."""
        target_probability = self.probabilities[target_id]
        return float(-np.log(max(target_probability, 1e-12)))

    def save_model(self, path=None, name="rnn_model"):
        """Save the model's name, learned weights, and biases."""
        if path is None:
            path = Path("checkpoints") / f"{name}.npz"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            name=name,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            lr=self.lr,
            Wxh=self.Wxh,
            Whh=self.Whh,
            Why=self.Why,
            bh=self.bh,
            by=self.by,
        )

        print(f"\nModel '{name}' saved to {path}")

    @classmethod
    def load_model(cls, name="rnn_model", path=None):
        """Load a saved model by name or path."""
        if path is None:
            path = Path("checkpoints") / (name if name.endswith(".npz") else f"{name}.npz")
        else:
            path = Path(path)

        with np.load(path, allow_pickle=False) as data:
            model = cls(
                input_size=int(data["input_size"]),
                hidden_size=int(data["hidden_size"]),
                output_size=int(data["output_size"]),
                lr=float(data["lr"]),
            )

            model.Wxh = data["Wxh"].copy()
            model.Whh = data["Whh"].copy()
            model.Why = data["Why"].copy()
            model.bh = data["bh"].copy()
            model.by = data["by"].copy()

            model.name = str(data["name"]) if "name" in data else path.stem

        print(f"\nModel '{model.name}' loaded from {path}")
        return model
