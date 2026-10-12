from AI.RNN.tokenization.tokenizer import (
    word_tokenize,
    create_vocabulary
)
from embeddings import (
    one_hot_encode,
    create_embedding_matrix,
    positional_encoding
)
import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    floatmode="fixed",
    sign="+"
)


# =====================================
# Utility
# =====================================

num_headers = 1


def print_header(text):
    global num_headers
    print("\n\n" + "=" * 60)
    print(f"EXPERIMENT {num_headers}: {text}")
    print("=" * 60 + "\n")
    num_headers += 1


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


# =====================================
# Setup
# =====================================

print_header("SETUP")

text = "The cat sat on the mat."

tokens = word_tokenize(text)
vocab, token_to_id, id_to_token = create_vocabulary(tokens)
vocab_size = len(vocab)

print(f"Text:       {text}")
print(f"Tokens:     {tokens}")
print(f"Vocabulary: {vocab}")
print(f"Token IDs:  {token_to_id}")
print(f"Vocab size: {vocab_size}")


# =====================================
# Experiment 1: One-hot encoding
# =====================================

print_header("ONE-HOT ENCODING")

print("Each token is represented by a vector with one 1 and zeros everywhere else.\n")

for token in vocab:
    token_id = token_to_id[token]
    vector = one_hot_encode(token_id, vocab_size)

    print(f"{token:>5} -> {vector}")

print("\nOne-hot vectors identify tokens, but do not represent semantic meaning.")


# =====================================
# Experiment 2: The embedding matrix
# =====================================

print_header("THE EMBEDDING MATRIX")

embedding_dim = 4

embedding_matrix = create_embedding_matrix(
    vocab_size,
    embedding_dim
)

print("Embedding matrix E:")
print(embedding_matrix)

print(f"\nShape: {embedding_matrix.shape}")
print(f"Vocabulary size (V): {vocab_size}")
print(f"Embedding dimension (d): {embedding_dim}")

print("\nMathematical representation:")
print(f"E has shape ({vocab_size}, {embedding_dim})")

print("\nEach row represents one token.")
print("Each column represents one embedding dimension.")


# =====================================
# Experiment 3: Token IDs -> embeddings
# =====================================

print_header("TOKEN-TO-EMBEDDING LOOKUP")

for token in vocab:
    token_id = token_to_id[token]
    embedding = embedding_matrix[token_id]

    print(f"{token:>5} (ID {token_id}) -> {embedding}")

print("\nA token ID selects one row from E.")


# =====================================
# Experiment 4: One-hot multiplication vs lookup
# =====================================

print_header("HOW EMBEDDING LOOKUP WORKS")

token = "cat"
token_id = token_to_id[token]

# Method 1: One-hot vector multiplied by E
cat_one_hot = one_hot_encode(token_id, vocab_size)
embedding_from_one_hot = cat_one_hot @ embedding_matrix

# Method 2: Direct row lookup
embedding_from_lookup = embedding_matrix[token_id]

print(f"Token: {token}")
print(f"Token ID: {token_id}")

print("\nOne-hot vector:")
print(cat_one_hot)

print("\nOne-hot @ embedding matrix:")
print(embedding_from_one_hot)

print("\nDirect embedding lookup:")
print(embedding_from_lookup)

print("\nAre the results equal?")
print(np.allclose(
    embedding_from_one_hot,
    embedding_from_lookup
))

print("\nDirect lookup is more efficient.")
print("One-hot multiplication illustrates the underlying mathematics.")


# =====================================
# Experiment 5: Embeddings are initially random
# =====================================

print_header("INITIAL EMBEDDINGS HAVE NO GUARANTEED MEANING")

cat_embedding = embedding_matrix[token_to_id["cat"]]
mat_embedding = embedding_matrix[token_to_id["mat"]]

similarity = cosine_similarity(
    cat_embedding,
    mat_embedding
)

print("Initial cosine similarity:")
print(f"cat vs mat: {similarity:+.4f}")

print("\nThese embeddings were randomly initialized.")
print("Their values do not automatically encode meaning.")
print("Training must adjust the vectors using a learning objective.")


# =====================================
# Experiment 6: Embeddings in a sequence
# =====================================

print_header("EMBEDDINGS IN A SEQUENCE")

sentence = ["the", "cat", "sat"]

for position, token in enumerate(sentence):
    token_id = token_to_id[token]
    embedding = embedding_matrix[token_id]

    print(f"Position {position}: {token}")
    print(f"{'Embedding:':<12}{embedding}\n")

print("Each token has its own embedding.")
print("The embedding lookup alone does not encode position.")


# =====================================
# Experiment 7: Positional encoding
# =====================================

print_header("POSITIONAL ENCODING (TRANSFORMER PREVIEW)")

# go through tokens, get position vector and token vector
for position, token in enumerate(sentence):
    token_id = token_to_id[token]

    # token vector
    embedding = embedding_matrix[token_id]

    # position vector
    position_vector = positional_encoding(
        position,
        embedding_dim
    )

    # combines both into ONE vector
    combined = embedding + position_vector

    print(f"Token: {token} (position {position})")
    print(f"{'Embedding:':<12}{embedding}")
    print(f"{'Position:':<12}{position_vector}  <-- sin/cos(pos / 10000^(2i/d))")
    print(f"{'Combined:':<12}{combined}\n")

print("Even dimensions use sin(); odd dimensions use cos() --> [even, odd, even, odd, ...]")
print("Adding positional encoding gives the embedding information about token position.")
print("RNNs normally represent sequence order through recurrent hidden states.")