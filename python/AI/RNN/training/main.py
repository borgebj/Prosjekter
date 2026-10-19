import numpy as np
import random
from pathlib import Path

from model.rnn import RNN
from tokenization.tokenizer import create_vocabulary, BPETokenizer, tokenize_corpus
from embedding.embeddings import create_embedding_matrix

# =====================================
# Configuration
# =====================================

embedding_dim = 8
hidden_size = 16
learning_rate = 0.01
num_merges = 50
epochs = 200

# Make experiments reproducible.
random.seed(42)
np.random.seed(42)

np.set_printoptions(
    precision=3,
    suppress=True,
    floatmode="fixed",
    sign="+"
)

# =====================================
# Load and Split Corpus
# =====================================

corpus_path = Path(__file__).with_name("corpus.txt")

corpus = [
    line.strip()
    for line in corpus_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

random.shuffle(corpus)

split_index = int(len(corpus) * 0.8)
train_corpus = corpus[:split_index]
test_corpus = corpus[split_index:]

# =====================================
# Tokenization and Vocabulary
# =====================================

bpe_tokenizer = BPETokenizer()
bpe_tokenizer.train(train_corpus, num_merges=num_merges)

train_tokenized = tokenize_corpus(train_corpus, bpe_tokenizer)
test_tokenized = tokenize_corpus(test_corpus, bpe_tokenizer)

train_tokens = [
    token
    for sentence in train_tokenized
    for token in sentence
]

vocab, token_to_id, id_to_token = create_vocabulary(train_tokens)
vocab_size = len(vocab)

unk_id = token_to_id["<UNK>"]


def encode_sentences(sentences):
    """Convert tokens to IDs, using UNK for unseen tokens."""
    return [
        [token_to_id.get(token, unk_id) for token in sentence]
        for sentence in sentences
    ]


train_token_id_sequences = encode_sentences(train_tokenized)
test_token_id_sequences = encode_sentences(test_tokenized)

# =====================================
# Embeddings
# =====================================

embedding_matrix = np.asarray(
    create_embedding_matrix(vocab_size, embedding_dim),
    dtype=np.float64
)

# =====================================
# RNN Initialization
# =====================================

model = RNN(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    output_size=vocab_size,
    lr=learning_rate
)

# =====================================
# Dataset and Model Summary
# =====================================

print("=" * 60)
print("RNN TRAINING - DATA AND MODEL SETUP")
print("=" * 60)

print("\nTraining corpus:")
print(*train_corpus[:7], sep="\n")

print(f"\nTraining sentences: {len(train_corpus)}")
print(f"Testing sentences:  {len(test_corpus)}")
print(f"Training tokens:    {len(train_tokens)}")
print(f"Vocabulary size:    {vocab_size}")
print(f"Embedding dim:      {embedding_dim}")
print(f"Hidden size:        {hidden_size}")
print(f"Learning rate:      {learning_rate}")
print(f"BPE merge rules:    {len(bpe_tokenizer.merges)}")

print("\nVocabulary:")
print(vocab[:20], "..." if len(vocab) > 20 else "")

# Flatten training token IDs for display.
token_ids = [
    token_id
    for sequence in train_token_id_sequences
    for token_id in sequence
]

print("\nTraining token IDs (first 30):")
print(token_ids[:30])

print("\nTraining tokens (first 30):")
print([id_to_token[i] for i in token_ids[:30]])

print("\nEmbedding Matrix (first 15 tokens):")
for token, token_id in list(token_to_id.items())[:15]:
    print(f"  {token:>8} -> {embedding_matrix[token_id]}")

# =====================================
# Create Next-Token Training Examples
# =====================================

training_examples = []

for sentence_ids in train_token_id_sequences:
    # Each target is the next token in the same sentence.
    for i in range(1, len(sentence_ids)):
        input_ids = sentence_ids[:i]
        target_id = sentence_ids[i]

        # Convert input token IDs into embedding vectors.
        x_sequence = embedding_matrix[input_ids]

        training_examples.append((x_sequence, target_id))

if not training_examples:
    raise ValueError(
        "No training examples created. "
        "Check that the training corpus contains sentences "
        "with at least two tokens."
    )

print(f"\nTraining examples: {len(training_examples)}")

# =====================================
# Training Loop
# =====================================

epoch_losses = []

for epoch in range(1, epochs + 1):
    random.shuffle(training_examples)
    total_loss = 0.0

    for x_sequence, target_id in training_examples:
        # Forward pass: calculate predictions.
        model.forward(x_sequence)

        # Calculate loss before updating weights.
        loss = model.compute_loss(target_id)
        total_loss += loss

        # Backward pass: calculate gradients and update weights.
        model.backward(target_id)

    average_loss = total_loss / len(training_examples)
    epoch_losses.append(average_loss)

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d}/{epochs}"
            f" | Average loss: {average_loss:.4f}"
        )

# =====================================
# Test Evaluation
# =====================================

test_loss = 0.0
test_examples = 0

for sentence_ids in test_token_id_sequences:
    for i in range(1, len(sentence_ids)):
        input_ids = sentence_ids[:i]
        target_id = sentence_ids[i]

        x_sequence = embedding_matrix[input_ids]

        # Forward pass only; do not update the model.
        model.forward(x_sequence)
        test_loss += model.compute_loss(target_id)
        test_examples += 1

average_test_loss = (
    test_loss / test_examples
    if test_examples > 0
    else float("nan")
)

# =====================================
# Training Results
# =====================================

print("\nTraining results")
print(f"Initial training loss: {epoch_losses[0]:.4f}")
print(f"Final training loss:   {epoch_losses[-1]:.4f}")
print(f"Training loss change:  {epoch_losses[-1] - epoch_losses[0]:+.4f}")
print(f"Test loss:             {average_test_loss:.4f}")
print(f"Test examples:         {test_examples}")


def generate_text(start_text, num_tokens):
    tokens = bpe_tokenizer.tokenize(start_text)
    token_ids = [token_to_id.get(token, unk_id) for token in tokens]

    for _ in range(num_tokens):
        x_sequence = embedding_matrix[token_ids]
        prediction_id, _ = model.predict(x_sequence)
        token_ids.append(prediction_id)

    generated_tokens = [id_to_token[i] for i in token_ids]

    # Reconstruct word boundaries from the ▁ marker.
    generated_text = "".join(generated_tokens).replace("▁", " ").strip()

    return generated_text


print("\nGenerated text:")
print(generate_text("the dog", num_tokens=15))
