import numpy as np


num_headers = 0


def print_header(text):
    global num_headers
    print("\n\n" + "=" * 60)
    print(f"EXPERIMENT {num_headers}: {text}")
    print("=" * 60 + "\n")
    num_headers += 1


# =====================================
# Experiment 0: RNN cell
# =====================================

print_header("RNN CELL")

# Our current input is the embedding for "the".
x = np.array([0.5, 0.2])

# Hidden state from the previous timestep.
h_previous = np.array([0.1, -0.3])

# Input → hidden weights.
W_x = np.array([
    [0.5, 0.2],
    [0.1, 0.4]
])

# Hidden → hidden weights.
W_h = np.array([
    [0.3, 0.1],
    [0.2, 0.5]
])

# Bias.
b = np.array([0.1, 0.1])


print("Input embedding for: \"the\"", x)
print("Previous hidden state:", h_previous)

# Contribution from the current input.
input_contribution = W_x @ x

# Contribution from the previous hidden state.
hidden_contribution = W_h @ h_previous

print("\nInput contribution:", input_contribution)
print("Hidden contribution:", hidden_contribution)

# Combine both contributions and the bias.
pre_activation = (
    input_contribution
    + hidden_contribution
    + b
)

print("Pre-activation:", pre_activation)

# Activate the pre-activation and create the new hidden state.
h = np.tanh(pre_activation)

print("New hidden state:", h)


# =====================================
# Experiment 1: RNN sequence
# - Forward pass through the sentence
# =====================================

print_header("RNN SEQUENCE")

# Tiny vocabulary for our example.
vocabulary = {
    0: "the",
    1: "cat",
    2: "sat",
    3: "on",
    4: "mat"
}

# Embeddings for the three input words.
#
# "the" → x_1
# "cat" → x_2
# "sat" → x_3
x_sequence = np.array([
    [0.5, 0.2],   # the
    [0.1, 0.7],   # cat
    [0.8, 0.3]    # sat
])

input_tokens = ["the", "cat", "sat"]

# Start with an empty hidden state.
h = np.array([0.0, 0.0])

print("Input sentence: \"the cat sat\"")
print("\nInput embeddings:")

for token, embedding in zip(input_tokens, x_sequence):
    print(f"{token:>4} -> {embedding}")

print("\nInitial hidden state: h_0 =", h)


# Process one word at a time.
for i, (token, x) in enumerate(zip(input_tokens, x_sequence)):

    # Current input contribution.
    input_contribution = W_x @ x

    # Previous memory contribution.
    hidden_contribution = W_h @ h

    # Combine current input + previous memory + bias.
    pre_activation = (
        input_contribution
        + hidden_contribution
        + b
    )

    # Create the new hidden state.
    h = np.tanh(pre_activation)

    print(f"\nTime step {i + 1}: \"{token}\"")
    print("Input:", x)
    print("Input contribution:", input_contribution)
    print("Previous hidden contribution:", hidden_contribution)
    print(f"New hidden state: h_{i + 1} =", h)


# =====================================
# Experiment 2: RNN output
# - Predict the next token
# =====================================

print_header("RNN OUTPUT")

# The correct next token in our example.
target_token_id = 3
target_token = vocabulary[target_token_id]

print("Input sentence: \"the cat sat\"")
print("Correct next token:", target_token)


# Output weights.
#
# 2 hidden dimensions
# → 5 possible output tokens
W_y = np.array([
    [0.5, 0.2],   # the
    [0.1, 0.4],   # cat
    [0.3, 0.6],   # sat
    [0.2, 0.1],   # on
    [0.4, 0.3]    # mat
])

# Output bias.
b_y = np.array([
    0.1,
    0.1,
    0.1,
    0.1,
    0.1
])


# The final hidden state represents the sequence
# processed so far: "the cat sat".
print("\nFinal hidden state:", h)


# Convert the hidden state into one score for each
# possible next token.
output_scores = W_y @ h + b_y

print("\nOutput scores:")

for token_id, score in enumerate(output_scores):
    print(f"{vocabulary[token_id]:>4} -> {score:.4f}")


# =====================================
# Softmax
# =====================================

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


probabilities = softmax(output_scores)

print("\nOutput probabilities:")

for token_id, probability in enumerate(probabilities):
    print(f"{vocabulary[token_id]:>4} -> {probability:.4f}")


print("\nSum of probabilities:", np.sum(probabilities))


# Select the token with the highest probability.
prediction_id = np.argmax(probabilities)
prediction_token = vocabulary[prediction_id]

print("\nPredicted token:", prediction_token)
print("Correct token:  ", target_token)