
from AI.RNN.tokenization.tokenizer import (
    word_tokenize,
    create_vocabulary,
)
from embeddings import (
    one_hot_encode,
    create_embedding_matrix,
    positional_encoding,
)
import numpy as np


np.set_printoptions(
    precision=3,
    suppress=True,
    floatmode="fixed",
    sign="+",
)


# =====================================
# Utility
# =====================================

num_headers = 0


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
# Setup: Reuse Block 1
# =====================================

print_header("TOKENIZATION AND VOCABULARY")

text = "The cat sat on the mat."

# Block 1: Convert text into tokens and IDs.
tokens = word_tokenize(text)
vocab, token_to_id, id_to_token = create_vocabulary(tokens)

token_ids = [token_to_id[token] for token in tokens]
vocab_size = len(vocab)

print(f"Text:       {text}")
print(f"Tokens:     {tokens}")
print(f"Vocabulary: {vocab}")
print(f"Token IDs:  {token_ids}")
print(f"Vocab size: {vocab_size}")

# Block 2 starts here: Convert token IDs into embeddings.


# =====================================
# Experiment 1: One-hot encoding
# =====================================

print_header("ONE-HOT ENCODING")

print("Each token is represented by a vector")
print("containing one 1 and zeros everywhere else.\n")

for token in vocab:
    token_id = token_to_id[token]
    vector = one_hot_encode(token_id, vocab_size)

    print(f"{token:>5} (ID {token_id}) -> {vector.astype(int)}")

print("\nOne-hot vectors identify tokens, but do not")
print("represent semantic relationships between them.")


# =====================================
# Experiment 2: The embedding matrix
# =====================================

print_header("THE EMBEDDING MATRIX")

print("NB! values are initially random\nTraining adjusts them to help learn representations.\n")

embedding_dim = 4

embedding_matrix = create_embedding_matrix(
    vocab_size,
    embedding_dim,
)

print("Embedding matrix E:")
print(embedding_matrix)

print(f"\nShape: {embedding_matrix.shape}")
print(f"Vocabulary size (V): {vocab_size}")
print(f"Embedding dimension (d): {embedding_dim}")

print(f"\nE has shape ({vocab_size}, {embedding_dim}).")
print("Each row represents one token.")
print("Each column represents one embedding dimension.")


# =====================================
# Experiment 3: Token IDs -> embeddings
# =====================================

print_header("TOKEN-TO-EMBEDDING LOOKUP")

for token, token_id in zip(tokens, token_ids):
    embedding = embedding_matrix[token_id]

    print(f"{token:>5} (ID {token_id}) -> {embedding}")

print("\nEach token ID selects a row from E.")
print("Repeated tokens use the same embedding row.")


# =====================================
# Experiment 4: One-hot multiplication vs lookup
# =====================================

print_header("HOW EMBEDDING LOOKUP WORKS")

token = "cat"
token_id = token_to_id[token]

# Method 1: Multiply the one-hot vector by E.
token_one_hot = one_hot_encode(token_id, vocab_size)
embedding_from_one_hot = token_one_hot @ embedding_matrix

# Method 2: Select the corresponding row directly.
embedding_from_lookup = embedding_matrix[token_id]

print(f"Token: {token}")
print(f"Token ID: {token_id}")

print("\nOne-hot vector:")
print(token_one_hot)

print("\nOne-hot @ embedding matrix:")
print(embedding_from_one_hot)

print("\nDirect embedding lookup:")
print(embedding_from_lookup)

print("\nAre the results equal?")
print(np.allclose(
    embedding_from_one_hot,
    embedding_from_lookup,
))

print("\nDirect lookup avoids constructing a one-hot vector.")
print("Both methods produce the same embedding.")


# =====================================
# Experiment 5: Initially random embeddings
# =====================================

print_header("EMBEDDINGS BEFORE TRAINING")

cat_embedding = embedding_matrix[token_to_id["cat"]]
mat_embedding = embedding_matrix[token_to_id["mat"]]

similarity = cosine_similarity(
    cat_embedding,
    mat_embedding,
)

print(f"Initial cosine similarity (cat vs mat): {similarity:+.4f}")

print("\nThe embeddings were randomly initialized.")
print("Their values do not yet encode learned semantic meaning.")
print("Training adjusts them using a learning objective.")


# =====================================
# Experiment 6: Embeddings in a sequence
# =====================================

print_header("EMBEDDINGS IN A SEQUENCE")

sentence = ["the", "cat", "sat"]
sentence_ids = [token_to_id[token] for token in sentence]

# Retrieve one embedding for each token, preserving order.
embedding_sequence = embedding_matrix[sentence_ids]

for position, (token, token_id, embedding) in enumerate(
    zip(sentence, sentence_ids, embedding_sequence)
):
    print(f"Position {position}:\t'{token}'\t(ID {token_id})")
    print(f"{'Embedding:':<12}{embedding}\n")

print(f"Embedding sequence shape: {embedding_sequence.shape}")
print("(Sequence length, embedding dimension)")

print("\nThe lookup preserves token order, but the individual")
print("embedding vectors do not explicitly encode position.")


# =====================================
# Experiment 7: Positional encoding
# =====================================

print_header("POSITIONAL ENCODING: TRANSFORMER PREVIEW")

print("Positional encoding adds information about token position.")
print("This example uses the same sentence and embedding matrix.\n")

for position, (token, token_id) in enumerate(
    zip(sentence, sentence_ids)
):
    # Token vector
    embedding = embedding_matrix[token_id]

    # Position vector
    position_vector = positional_encoding(
        position,
        embedding_dim,
    )

    # Combine token and position information.
    combined = embedding + position_vector

    print(f"Token:\t'{token}'\t(position {position})")
    print(f"{'Embedding:':<12}{embedding}")
    print(f"{'Position:':<12}{position_vector}")
    print(f"{'Combined:':<12}{combined}\n")

print("The position vector is added to the token embedding.")
print("Sinusoidal positional encoding uses sine and cosine functions.")

print("\nImportant distinction:")
print("Transformers commonly use positional encodings.")
print("RNNs normally represent sequence order through recurrent")
print("hidden states rather than requiring this positional encoding.")
