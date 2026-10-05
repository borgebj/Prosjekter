import numpy as np


# =====================================
# Textual representations

def one_hot_encode(token_id, vocab_size):
    """Creates a onehot-encoded vector for 'token_id'"""
    vector = np.zeros(vocab_size)  # =  '[0] * vocab_size'
    vector[token_id] = 1
    return vector


def create_embedding_matrix(vocab_size, embedding_dim):
    """Creates a random embedding matrix"""

    # creates a random matrix
    embedding_matrix = np.random.uniform(
        -1, 1,
        size=(vocab_size, embedding_dim)
    )

    return embedding_matrix
