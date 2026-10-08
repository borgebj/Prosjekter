from AI.RNN.tokenization.tokenizer import (
    word_tokenize,
    create_vocabulary
)
from embeddings import (
    one_hot_encode,
    create_embedding_matrix
)
from embedding_trainer import EmbeddingTrainer, cosine_similarity
import numpy as np

# =====================================
# Utility for testing


num_headers = 1


def print_header(text):
    global num_headers
    print("\n\n" + "=" * 60)
    print(f"EXPERIMENT {num_headers}: {text}")
    print("=" * 60, "\n")
    num_headers += 1


def dot_product(a, b):
    """Simple dot product of two vectors"""
    return sum(x * y for x, y in zip(a, b))


def print_similarities(embedding_matrix, token_to_id):
    comparisons = [
        ("cat", "dog"),
        ("car", "bus"),
        ("cat", "car"),
        ("dog", "bus")
    ]

    for word1, word2 in comparisons:
        a = embedding_matrix[token_to_id[word1]]
        b = embedding_matrix[token_to_id[word2]]

        similarity = cosine_similarity(a, b)

        print(f"{word1} - {word2}: {similarity * 100:.2f}%")


def print_embeddings():
    for word in words:
        word_id = token_to_id[word]
        print(f"{word} ---> {embedding_matrix[word_id]}")


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

for token in vocab:
    token_id = token_to_id[token]
    vector = one_hot_encode(token_id, vocab_size)

    print(f"{token:>5} (ID {token_id}) -> {vector}")

# =====================================
# Experiment 2: One-hot vector properties
# =====================================
print_header("ONE-HOT VECTOR PROPERTIES")

cat = one_hot_encode(token_to_id["cat"], vocab_size)
mat = one_hot_encode(token_to_id["mat"], vocab_size)
sat = one_hot_encode(token_to_id["sat"], vocab_size)

print("Same gives 1, different gives 0")
print(f"cat · cat = {dot_product(cat, cat)}")
print(f"cat · mat = {dot_product(cat, mat)}")
print(f"cat · sat = {dot_product(cat, sat)}")
print(f"mat · sat = {dot_product(mat, sat)}")

print("\nDifferent tokens have dot product 0.")
print("The same token has dot product 1.")

# =====================================
# Experiment 3: One-hot limitation
# =====================================
print_header("ONE-HOT LIMITATION")

print("One-hot vectors only tell us whether tokens are identical.")
print("They contain no information about semantic similarity.\n")

print(f"cat: {cat}")
print(f"mat: {mat}")

print(f"\ncat · mat = {dot_product(cat, mat)}")
print("→ No relationship is represented between the words.")

# =====================================
# Experiment 4: One-hot matrix
# =====================================
print_header("ONE-HOT MATRIX")

one_hot_matrix = []

for token in vocab:
    token_id = token_to_id[token]
    vector = one_hot_encode(token_id, vocab_size)
    one_hot_matrix.append(vector)

print("Each row corresponds to one vocabulary token:\n")

for token, vector in zip(vocab, one_hot_matrix):
    print(f"{token:>5}: {vector}")


# =====================================
# Experiment 5: Embedding matrix
# =====================================
print_header("EMBEDDING MATRIX")
print("The embedding matrix assigns each token in the vocabulary to a unique vector [x1, x2, x3<]\n")

embedding_dim = 3

embedding_matrix = create_embedding_matrix(
    vocab_size,
    embedding_dim
)

print("Embedding matrix:")
print(embedding_matrix)

print(f"\nShape: {embedding_matrix.shape}")
print(f"Rows:    {vocab_size} tokens")
print(f"Columns: {embedding_dim} embedding dimensions")

# =====================================
# Experiment 6: Token → embedding
# =====================================
print_header("TOKEN -> EMBEDDING")

for token in vocab:
    # gets token-id  'cat' -> '1'
    token_id = token_to_id[token]

    # gets embedding from id  '1' -> '[0,1, -0.2, 0.9]
    embedding = embedding_matrix[token_id]

    print(f"{token:>5} (ID {token_id}) -> {embedding}")

# =====================================
# Experiment 7: One-hot → embedding
# =====================================
print_header("ONE-HOT x EMBEDDING MATRIX")

# creates onehot for 'cat'
cat_one_hot = one_hot_encode(
    token_to_id["cat"],
    vocab_size
)

# selects embedding-row using onehot  (due to 0s and 1s)
cat_embedding = cat_one_hot @ embedding_matrix

print(f"cat one-hot:  {cat_one_hot}")
print(f"embedding:    {cat_embedding}")

# =====================================
# Experiment 8: Direct lookup
# =====================================
print_header("DIRECT EMBEDDING LOOKUP")

# gets 'cat ID for lookup
cat_id = token_to_id["cat"]

# gets 'cat' embedding from ID - '0' -> '[0.1, -0.3, 0.9]
cat_embedding_lookup = embedding_matrix[cat_id]

print(f"cat ID:       {cat_id}")
print(f"embedding:    {cat_embedding_lookup}")

# =====================================
# Experiment 9: Compare both methods
# =====================================
print_header("COMPARE EMBEDDING METHODS")

print("One-hot × embedding matrix:")
print(cat_embedding)

print("\nDirect matrix lookup:")
print(cat_embedding_lookup)

print("\nAre they equal?")
print(np.array_equal(cat_embedding, cat_embedding_lookup))

# =====================================
# Experiment: Train embeddings
# =====================================
print_header("TRAIN EMBEDDINGS")

words = ["cat", "dog", "car", "bus"]

vocab, token_to_id, id_to_token = create_vocabulary(
    words,
    include_unk=False
)

embedding_dim = 2

embedding_matrix = create_embedding_matrix(
    len(vocab),
    embedding_dim
)


# =====================================
# Training data
# =====================================

# Pairs we want to move closer together.
similar_pairs = [
    ("cat", "dog"),
    ("car", "bus")
]

# Pairs we want to move apart.
different_pairs = [
    ("cat", "car"),
    ("cat", "bus"),
    ("dog", "car"),
    ("dog", "bus")
]


# =====================================
# Initial state
# =====================================

print("Initial embeddings:")
print_embeddings()

print("\nInitial similarities:")
print_similarities(embedding_matrix, token_to_id)


# =====================================
# Training
# =====================================

trainer = EmbeddingTrainer(
    embedding_matrix,
    token_to_id,
    similar_pairs,
    different_pairs
)

learning_rate = 0.05
epochs = 10

print(f"\nInitial loss: {trainer.calculate_loss():.4f}")
print("\nTraining:")
trainer.train(
    learning_rate,
    epochs
)


# =====================================
# Final state
# =====================================

print("\nFinal embeddings:")
print_embeddings()

print(f"\nFinal loss: {trainer.calculate_loss():.4f}")

print("\nFinal similarities:")
print_similarities(embedding_matrix, token_to_id)
print("\n" + "="*10)
print("Dot products: 0 means different, 1 means same")
print("cat · mat:", dot_product(cat, mat))
print("mat · sat:", dot_product(mat, sat))
print("mat · mat:", dot_product(mat, mat))
