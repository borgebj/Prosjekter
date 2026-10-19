import numpy as np


# =====================================
# Textual representations

def one_hot_encode(token_id, vocab_size):
    """Creates a onehot-encoded vector for 'token_id'"""
    vector = np.zeros(vocab_size)  # =  '[0] * vocab_size'
    vector[token_id] = 1
    return vector


def create_embedding_matrix(vocab_size, embedding_dim):
    """Creates a _random_ embedding matrix"""

    # creates a random matrix
    embedding_matrix = np.random.uniform(
        -1, 1,
        size=(vocab_size, embedding_dim)
    )

    return embedding_matrix


def positional_encoding(position, embedding_dim):
    """
    Generate a sinusoidal positional encoding for a token position.
    Alternates sine and cosine values to represent sequence position.
    """
    # [0, 0, 0, ..., 0]
    encoding = np.zeros(embedding_dim)

    # each dimension, aka "each list value""
    for i in range(embedding_dim):

        # position-dependent angle for this dimension
        angle = position / (
                10000 ** (2 * (i // 2) / embedding_dim)
        )
        # sin on even
        if i % 2 == 0:
            encoding[i] = np.sin(angle)

        # cos on odd
        else:
            encoding[i] = np.cos(angle)

    return encoding
