import rnn
from rnn import RNN
import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    floatmode="fixed",
    sign="+"
)

num_experiments = 1


def print_header(text):
    global num_experiments
    print("\n\n" + "=" * 60)
    print(f"EXPERIMENT {num_experiments}: {text}")
    print("=" * 60)
    num_experiments += 1


# =====================================
# Setup
# =====================================

vocabulary = {
    0: "the",
    1: "cat",
    2: "sat",
    3: "on",
    4: "mat"
}

# Shape: (sequence length, embedding dimension)
# 3 tokens, each represented by 2 numbers
x_sequence = np.array([
    [0.5, 0.2],  # Embedding for "the": 2 features
    [0.1, 0.7],  # Embedding for "cat": 2 features
    [0.8, 0.3]  # Embedding for "sat": 2 features
])

input_tokens = ["the", "cat", "sat"]

W_x = np.array([
    [0.5, 0.2],
    [0.1, 0.4]
])

W_h = np.array([
    [0.3, 0.1],
    [0.2, 0.5]
])

b = np.array([0.1, 0.1])

W_y = np.array([
    [0.5, 0.2],  # the
    [0.1, 0.4],  # cat
    [0.3, 0.6],  # sat
    [0.2, 0.1],  # on
    [0.4, 0.3]  # mat
])

b_y = np.full(5, 0.1)

print("\nEmbedding dimension: 2 values per token --> [f1, f2]")
print("Sequence length:     3 tokens --> 'the' 'cat' 'sat'")
print("Input shape:         (3, 2)")
print("Hidden state size:   2 values\n")
print(f"Embeddings:\n\t{x_sequence[0]}\t'the'\n\t{x_sequence[1]}\t'cat'\n\t{x_sequence[2]}\t'sat'")

# =====================================
# Experiment 1: RNN cell
# =====================================

print_header("RNN CELL")

# Embedding for token "the"
x = x_sequence[0]

# Initial hidden state: no previous context
h_previous = np.zeros(2)  # Two hidden-state values, both initialized to 0

# Transform the current token embedding using input-to-hidden weights
input_contribution = W_x @ x

# Transform the previous hidden state using recurrent weights
hidden_contribution = W_h @ h_previous

# Combine input, previous state, and bias before activation
pre_activation = input_contribution + hidden_contribution + b

# Apply tanh to calculate the new hidden state
h = np.tanh(pre_activation)

print("Input token:          'the'")
print("Embedding:           ", x)
print("Previous hidden:     ", h_previous, "\t-\t initially empty")
print("Input contribution:  ", input_contribution, "\t-\t input weights @ embedding")
print("Hidden contribution: ", hidden_contribution, "\t-\t hidden weights @ prev hidden state")
print("Pre-activation:      ", pre_activation, "\t-\t z = (input_contrib + hidden_contrib + bias)")
print("New hidden state:    ", h, "\t-\t tanh(z)")

# =====================================
# Experiment 2: Process a sequence
# =====================================

print_header("RNN SEQUENCE")

h = np.zeros(2)

print(f"Input sentence:       '{' '.join(input_tokens)}'")
print("Initial hidden state:", h)

for timestep, (token, x) in enumerate(
        zip(input_tokens, x_sequence), start=1
):
    h_previous = h.copy()

    # (input) weights @ embedding
    input_contribution = (W_x @ x)

    # (hidden) weights @ previous state
    hidden_contribution = (W_h @ h_previous)

    pre_activation = (
            input_contribution
            + hidden_contribution
            + b  # ias
    )

    # hidden activation
    h = np.tanh(pre_activation)

    print(f"\nTimestep {timestep}: [ '{token}' ]")
    print(f"(1)  Embedding:          \t {x} \t-\t embeddings[{timestep - 1}]")
    print(f"(2)  Input contribution: \t {input_contribution} \t-\t W_x @ x")
    print(f"(3)  Hidden contribution:\t {hidden_contribution} \t-\t W_h @ prev_h")
    print(f"(4)  Combined + bias:    \t {pre_activation} \t-\t input + hidden + bias")
    print(f"(5)  Hidden state:       \t {h} \t-\t tanh(combined)")

print("\nFinal hidden state:  ", h)

# =====================================
# Experiment 3: Output and prediction
# =====================================

print_header("OUTPUT AND PREDICTION")

target_token_id = 3
target_token = vocabulary[target_token_id]

output_scores = W_y @ h + b_y


def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


# softmax:
#   - converts to val between 0 and 1
#   - normalize so all so sum(all) = 1
probabilities = softmax(output_scores)

print(f"Input sentence:       '{' '.join(input_tokens)}'")
print(f"Correct next token:   '{target_token}'")

print("\nToken scores and probabilities:")

for token_id, token in vocabulary.items():
    print(
        f"  [{token:>4} ]: "
        f"score = {output_scores[token_id]:+.3f}  |  "
        f"probability = {probabilities[token_id]:.3f}"
    )

# highest probability among all
prediction_id = np.argmax(probabilities)
prediction_token = vocabulary[prediction_id]

print(f"\nPredicted token:     '{prediction_token}'")
print(f"Correct token:       '{target_token}'")

# =====================================
# Experiment 4: Cross-entropy loss
# =====================================

print_header("CROSS-ENTROPY LOSS")

target_probability = probabilities[target_token_id]
loss = -np.log(target_probability)

print(f"Correct token:         \t '{target_token}'")
print(f"Probability of target: \t {target_probability:.3f}")
print(f"Cross-entropy loss:    \t {loss:.3f}")



# =====================================
# Experiment 5: RNN model and weights
# =====================================

print_header("RNN MODEL AND WEIGHTS")

print("Xavier initialization scales random weights.")
print("Biases start at zero.\n")

model = RNN(
    input_size=2,
    hidden_size=2,
    output_size=len(vocabulary)
)

# Process the input sequence
hidden_states = model.forward(x_sequence)

print("W_x (input-to-hidden):")
print(model.W_x)

print("\nW_h (hidden-to-hidden):")
print(model.W_h)

print("\nW_y (hidden-to-output):")
print(model.W_y)

print("\nHidden states:\n", hidden_states)

# Predict the next token
prediction_id, probabilities = model.predict(x_sequence)

print(f"\nInput sequence:    '{' '.join(input_tokens)}'")
print(f"Target next word:  '{target_token}'")

print("\nNext-word probabilities:")
for token_id, token in vocabulary.items():
    print(f"  {token:>4}: {probabilities[token_id]:.3f}")

print(f"\nPredicted next word: '{vocabulary[prediction_id]}'")
print(f"Full prediction:     '{' '.join(input_tokens)} {vocabulary[prediction_id]}'")