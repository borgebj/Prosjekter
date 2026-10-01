
def one_hot_encode(token_id, vocab_size):
    """Creates a onehot-encoded vector for 'token_id'"""
    vector = [0] * vocab_size
    vector[token_id] = 1
    return vector


def dot_product(a, b):
    """Simple dot product of two vectors"""
    return sum(x * y for x,y in zip(a, b))