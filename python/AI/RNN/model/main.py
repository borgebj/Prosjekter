from rnn import RNN

from AI.RNN.tokenization.tokenizer import (
    word_tokenize,
    create_vocabulary,
)
from AI.RNN.embedding.embeddings import (
    create_embedding_matrix,
)
import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    floatmode="fixed",
    sign="+"
)

num_experiments = 0


def print_header(text):
    global num_experiments
    print("\n\n" + "=" * 60)
    print(f"EXPERIMENT {num_experiments}: {text}")
    print("=" * 60)
    num_experiments += 1


# =====================================
# Setup: Reuse Block 1 and Block 2
# =====================================

print_header("TOKENIZATION AND EMBEDDINGS")

text = "The cat sat on the mat."

# Block 1: Tokenize the text and create a vocabulary.
tokens = word_tokenize(text)
vocab, token_to_id, id_to_token = create_vocabulary(tokens)

token_ids = np.array([token_to_id[token] for token in tokens], dtype=np.int64)

vocabulary = id_to_token
vocab_size = len(vocab)

# Block 2: Convert token IDs into embedding vectors.
embedding_dim = 2

embedding_matrix = np.asarray(
    create_embedding_matrix(vocab_size, embedding_dim),
    dtype=np.float64
)

# Block 3: RNN initialization
# use all tokens except last as input
input_tokens = tokens[:-1]
input_ids = token_ids[:-1]

target_token = tokens[-1]  # final is target
target_token_id = token_to_id[target_token]

# look up embedding vector for each input token
# shape:  (n-tokens, embedding_dim)
x_sequence = embedding_matrix[input_ids]

sequence_length = len(input_tokens)

# RNN Configuration
hidden_size = 2

print(f"Text:                '{text}'")
print(f"Tokens:              {tokens}")
print(f"Vocabulary:          {vocab}")
print(f"Token IDs:           [{' '.join(map(str, token_ids))}] -> [{' '.join(tokens)}]")
print(f"Input tokens:        '{' '.join(input_tokens)}'")
print(f"Target next token:   '{target_token}'\n")
print("Input embeddings:")

for token, embedding in zip(input_tokens, x_sequence):
    print(f"  {token:>5} -> {embedding}")
print(f"  {target_token:>5} -> {embedding_matrix[target_token_id]}  (target)")

# =====================================
# Experiment 1: RNN Forward
# =====================================

print_header("RNN FORWARD")

# initialize model with random weights - no training yet
model = RNN(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    output_size=vocab_size
)

# PERFORMS A MANUAL FORWARD PASS FOR EXPERIMENT-SAKE
h_previous = np.zeros(hidden_size)
hidden_states = []

print(f"Initial hidden state: {h_previous}")

for timestep, (token, x) in enumerate(
        zip(input_tokens, x_sequence), start=1
):
    input_contribution = model.W_x @ x
    hidden_contribution = model.W_h @ h_previous

    pre_activation = (
            input_contribution
            + hidden_contribution
            + model.b
    )

    h = np.tanh(pre_activation)
    hidden_states.append(h.copy())

    print(f"\nTimestep {timestep}: [ '{token}' ]")
    print(f"(1) Embedding:           \t {x}")
    print(f"(2) Input contribution:  \t {input_contribution}")
    print(f"(3) Hidden contribution: \t {hidden_contribution}")
    print(f"(4) Combined + bias:     \t {pre_activation}")
    print(f"(5) Hidden state:        \t {h}")

    h_previous = h.copy()

hidden_states = np.array(hidden_states)

print("\nFinal hidden state:   \t\t", hidden_states[-1])


# =====================================
# Experiment 2: Output and prediction
# =====================================

print_header("OUTPUT AND PREDICTION")

# final hidden state summarize the processed input sequence
final_hidden_state = hidden_states[-1]

# retrieve scores and probabilities calculated during forward()
prediction_id, probabilities = model.predict(x_sequence)
output_scores = model.output_scores

print(f"Input sentence:       '{' '.join(input_tokens)}'")
print(f"Correct next token:   '{target_token}'")
print("Final hidden state:  ", final_hidden_state)
print("\nOutput scores and probabilities:")

for token_id, token in id_to_token.items():
    print(f"  [{token:>5} ]: "
          f"score = {output_scores[token_id]:+.3f}  |  "
          f"probability = {probabilities[token_id]:.3f} ({int(probabilities[token_id] * 100)}%)"
          )

# convert predicted token ID back to its token
prediction_token = id_to_token[prediction_id]

print(f"\nPredicted token:      '{prediction_token}'")
print(f"Correct token:        '{target_token}'")
print(f"Full prediction:      '{' '.join(input_tokens)} {prediction_token}'")


# =====================================
# Experiment 3: Cross-entropy loss
# =====================================

print_header("CROSS-ENTROPY LOSS")

# calculate loss using probability of correct next token
loss_before = model.compute_loss(target_token_id)
target_probability = probabilities[target_token_id]

print(f"Correct token:         \t '{target_token}'")
print(f"Target token ID:       \t {target_token_id}")
print(f"Target probability:    \t {target_probability:.3f} ({int(target_probability * 100)}%)")
print(f"Cross-entropy loss:    \t {loss_before:.3f}")

print("\nLoss measures how much probability the model")
print("assigns to the correct next token.")
print("The model has not been trained yet.")


# =====================================
# Experiment 4: Backpropagation
# =====================================

print_header("BACKPROPAGATION AND ONE TRAINING STEP")

# calculate gradients using BPTT and update weights
model.backward(target_token_id)

# recalculate prediction
prediction_id, probabilities = model.predict(x_sequence)
loss_after = model.compute_loss(target_token_id)

prediction_token = id_to_token[prediction_id]
target_probability_after = probabilities[target_token_id]

print(f"\nPredicted token:       '{prediction_token}'")
print(f"Correct token:         '{target_token}'")
print(f"Target probability:     {target_probability_after:.3f} ({target_probability_after*100:.2f}%) \t vs \t before ({target_probability*100:.2f}%)")
print(f"Loss after update:      {loss_after:.4f}")
print(f"Loss change:            {loss_after - loss_before:+.4f}")

if loss_after < loss_before: print("\nThe update reduced the loss.")
elif loss_after > loss_before: print("\nThe update increased the loss.")
else: print("\nThe loss remained unchanged.")


# =====================================
# Experiment 5: MODEL TRAINING
# =====================================

print_header("MODEL TRAINING")

losses = []

for _ in range(100):
    model.forward(x_sequence)
    losses.append(model.compute_loss(target_token_id))
    model.backward(target_token_id)

print(f"Loss, start of training: \t{losses[0]:.4f}")
print(f"Loss, end of training:   \t{losses[-1]:.4f}")

prediction_id, probabilities = model.predict(x_sequence)
prediction_token = id_to_token[prediction_id]
target_probability_trained = probabilities[target_token_id]

print(f"Predicted token:    \t\t'{prediction_token}'")
print(f"Target probability: \t\t{target_probability_trained:.3f} ({int(target_probability_trained * 100)}%)")
print(f"Correct token:      \t\t'{target_token}'")
print(f"Full prediction:     \t\t'{' '.join(input_tokens)} {prediction_token}'")