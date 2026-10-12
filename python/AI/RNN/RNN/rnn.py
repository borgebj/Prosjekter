import numpy as np


def xavier_init(fan_in:int, fan_out:int) -> np.ndarray:
    # Xavier (Glorot) normal initialization
    # - keeps activation variance stable across layers
    # - common with sigmoid/tanh activations

    std = (2 / (fan_in + fan_out)) ** 0.5
    weights = np.random.randn(fan_out, fan_in) * std

    return weights


class RNN:

    def __init__(self, input_size:int = 2, hidden_size:int =2):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # weight initialization

        # Input-to-hidden weight initialization
        self.W_x = xavier_init(input_size, hidden_size)

        # Hidde-to-hidden (recurrent) weight initialization
        self.W_h = xavier_init(hidden_size, hidden_size)

        # Hidden-state bias to zeros    ->   [0, 0, ...]
        self.b = np.zeros(hidden_size)

    def forward(self, x_sequence):

        # hidden state starts at 0
        h = np.zeros(self.hidden_size)
        hidden_states = []

        # for each embedding/token
        for x in x_sequence:
            # together w/ bias creates z (pre-act)
            input_contribution = self.W_x @ x
            hidden_contribution = self.W_h @ h

            # activation on Z (pre-act)
            h = np.tanh(input_contribution + hidden_contribution + self.b)

            hidden_states.append(h.copy())

        return np.array(hidden_states)
