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
        self.W_x = xavier_init(input_size, hidden_size)
        self.W_h = xavier_init(hidden_size, hidden_size)
        self.b = np.zeros(hidden_size)

        # Hidden-to-output weights and output bias.
        self.W_y = xavier_init(hidden_size, output_size)
        self.b_y = np.zeros(output_size)

        # Cache values from the most recent forward pass for BPTT.
        self.x_sequence = None
        self.hidden_states = None
        self.output_scores = None
        self.probabilities = None

    def forward(self, x_sequence: np.ndarray) -> np.ndarray:
        """Process a sequence and return output logits."""

        x_sequence = np.asarray(x_sequence, dtype=np.float64)
        self.x_sequence = x_sequence.copy()

        h = np.zeros(self.hidden_size)
        hidden_states = []

        # Process each input vector in order.
        for x in self.x_sequence:
            z = self.W_x @ x + self.W_h @ h + self.b
            h = np.tanh(z)
            hidden_states.append(h.copy())

        self.hidden_states = np.asarray(hidden_states)

        # Generate output logits from the final hidden state.
        final_hidden_state = self.hidden_states[-1]
        self.output_scores = self.W_y @ final_hidden_state + self.b_y

        self.probabilities = softmax(self.output_scores)

        return self.output_scores.copy()

    def predict(
            self,
            x_sequence: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Return the predicted class ID and its probability distribution."""

        self.forward(x_sequence)

        prediction_id = int(np.argmax(self.probabilities))

        return prediction_id, self.probabilities.copy()

    def compute_loss(self, target_id: int) -> float:
        """Calculate cross-entropy loss for the target class."""

        target_probability = self.probabilities[target_id]
        return float(-np.log(max(target_probability, 1e-12)))

    def backward(self, target_id: int) -> None:
        """Calculate gradients using BPTT and update model parameters."""

        # Output layer gradients: softmax + cross-entropy.
        d_scores = self.probabilities.copy()
        d_scores[target_id] -= 1.0

        final_hidden_state = self.hidden_states[-1]

        dW_y = np.outer(d_scores, final_hidden_state)
        db_y = d_scores.copy()

        # Propagate output error into the final hidden state.
        dh = self.W_y.T @ d_scores

        dW_x = np.zeros_like(self.W_x)
        dW_h = np.zeros_like(self.W_h)
        db = np.zeros_like(self.b)

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

            dW_x += np.outer(dz, x)
            dW_h += np.outer(dz, h_previous)
            db += dz

            # Propagate error to the previous hidden state.
            dh = self.W_h.T @ dz

        # Gradient clipping limits the size of large updates.
        gradients = [dW_x, dW_h, db, dW_y, db_y]

        grad_norm = np.sqrt(
            sum(np.sum(gradient ** 2) for gradient in gradients)
        )

        if (
                self.max_grad_norm is not None
                and grad_norm > self.max_grad_norm
        ):
            scale = self.max_grad_norm / (grad_norm + 1e-12)
            gradients = [gradient * scale for gradient in gradients]

        dW_x, dW_h, db, dW_y, db_y = gradients

        # SGD (Stochastic Gradient Descent): update weights using
        # the gradients from this training example to reduce loss.
        self.W_x -= self.lr * dW_x
        self.W_h -= self.lr * dW_h
        self.b -= self.lr * db

        self.W_y -= self.lr * dW_y
        self.b_y -= self.lr * db_y
