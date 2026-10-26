"""
Block 3: RNN showcase.

Run this file to follow a tiny RNN through next-token prediction:
forward pass, prediction and loss, then training. The tokens and
embeddings come from Blocks 1 and 2, and all experiments share one
sentence and one model (see Context).

Note: the model is created once and shared. The training experiment
changes its weights, so it must stay last.
"""
from AI.RNN.showcase import Showcase, header, takeaway, bolden
from AI.RNN.model.rnn import RNN
from AI.RNN.tokenization.tokenizer import word_tokenize, create_vocabulary
from AI.RNN.embedding.embeddings import create_embedding_matrix
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    floatmode="fixed",
    sign="+",
)

TEXT = "The cat sat on the mat."
EMBEDDING_DIM = 2
HIDDEN_SIZE = 2
TRAINING_STEPS = 100

showcase = Showcase()


# =====================================
# Shared data (Blocks 1 and 2 reused)
# =====================================

class Step(NamedTuple):
    """Every intermediate value of one RNN timestep."""
    token: str
    x: np.ndarray                # embedding of the current token
    input_part: np.ndarray       # W_x @ x
    hidden_part: np.ndarray      # W_h @ h_previous
    pre_activation: np.ndarray   # input_part + hidden_part + b
    h: np.ndarray                # tanh(pre_activation): the new hidden state


@dataclass
class Context:
    """Data shared by every experiment."""
    tokens: list[str]
    vocab: list[str]
    id_to_token: dict[int, str]
    token_ids: list[int]
    input_tokens: list[str]          # every token except the last
    target_token: str                # the last token: what the model should predict
    target_token_id: int
    embedding_matrix: np.ndarray     # E: (vocab size, embedding dim)
    x_sequence: np.ndarray           # embeddings of the input tokens: (n inputs, embedding dim)
    model: RNN                       # random weights, changed by the training experiment
    steps: list[Step]                # manual forward pass of the untrained model


def forward_steps(model: RNN, input_tokens: list[str], x_sequence: np.ndarray) -> list[Step]:
    """Runs the forward pass by hand, keeping every intermediate value."""
    h = np.zeros(HIDDEN_SIZE)
    steps = []

    for token, x in zip(input_tokens, x_sequence):
        input_part = model.Wxh @ x
        hidden_part = model.Whh @ h
        pre_activation = input_part + hidden_part + model.bh
        h = np.tanh(pre_activation)
        steps.append(Step(token, x, input_part, hidden_part, pre_activation, h))

    return steps


def build_context() -> Context:
    # Block 1: text -> tokens -> IDs
    tokens = word_tokenize(TEXT)
    vocab, token_to_id, id_to_token = create_vocabulary(tokens)
    token_ids = [token_to_id[t] for t in tokens]

    # Block 2: IDs -> embeddings
    embedding_matrix = np.asarray(
        create_embedding_matrix(len(vocab), EMBEDDING_DIM),
        dtype=np.float64,
    )

    # Block 3: use every token but the last as input, the last as target
    input_tokens, target_token = tokens[:-1], tokens[-1]
    input_ids, target_token_id = token_ids[:-1], token_ids[-1]
    x_sequence = embedding_matrix[input_ids]

    model = RNN(
        input_size=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        output_size=len(vocab),
    )

    return Context(
        tokens=tokens,
        vocab=vocab,
        id_to_token=id_to_token,
        token_ids=token_ids,
        input_tokens=input_tokens,
        target_token=target_token,
        target_token_id=target_token_id,
        embedding_matrix=embedding_matrix,
        x_sequence=x_sequence,
        model=model,
        steps=forward_steps(model, input_tokens, x_sequence),
    )


def show_setup(ctx: Context) -> None:
    header("SETUP: TOKENS, EMBEDDINGS AND TARGET (Blocks 1 and 2)")
    print(f"Text:               '{TEXT}'")
    print(f"Tokens:             {ctx.tokens}")
    print(f"Vocabulary:         {ctx.vocab}")
    print(f"Token IDs:          {ctx.token_ids}")
    print(f"Input tokens:       '{' '.join(ctx.input_tokens)}'")
    print(f"Target next token:  '{ctx.target_token}'\n")

    print("Embeddings:")
    for token, embedding in zip(ctx.input_tokens, ctx.x_sequence):
        print(f"  {token:>5} -> {embedding}")

    print(bolden(f"\n  {ctx.target_token:>5} -> {ctx.embedding_matrix[ctx.target_token_id]}  (target)"))


# =====================================
# Experiments
# =====================================

@showcase.experiment
def rnn_forward(ctx: Context) -> None:
    """RNN forward pass

    The RNN reads the input tokens one at a time. At every timestep it combines
    the current embedding with the previous hidden state into a new hidden state:
        h = tanh(W_x @ x + W_h @ h_previous + b)
    The weights are random, because the model is not trained yet.
    """
    print(f"Initial hidden state: {np.zeros(HIDDEN_SIZE)}")

    for timestep, step in enumerate(ctx.steps, start=1):
        print(f"\nTimestep {timestep}: '{step.token}'")
        print(f"  (1) Embedding:           {step.x}")
        print(f"  (2) Input contribution:  {step.input_part} \t-  input weights @ embedding")
        print(f"  (3) Hidden contribution: {step.hidden_part} \t-  hidden weights @ previous hidden state")
        print(f"  (4) Combined + bias:     {step.pre_activation} \t-  input + hidden + bias")
        print(f"  (5) Hidden state:        {step.h} \t-  tanh(combined)")

    print(f"\nFinal hidden state: {ctx.steps[-1].h} (used for prediction)")

    takeaway(
        "The final hidden state summarizes the whole input sequence.",
        "The output layer uses it to predict the next token.",
    )


@showcase.experiment
def prediction_and_loss(ctx: Context) -> None:
    """From hidden state to prediction and loss

    The output layer turns the final hidden state into one score per vocabulary token,
    and softmax converts the scores into probabilities.
    Cross-entropy loss measures how much probability went to the correct token (-log p).
    """
    model = ctx.model
    prediction_id, probabilities = model.predict(ctx.x_sequence)
    scores = model.output_scores
    loss = model.compute_loss(ctx.target_token_id)

    print(f"Input:               '{' '.join(ctx.input_tokens)}'")
    print(f"Correct next token:  '{ctx.target_token}'")
    print(f"Final hidden state:  {ctx.steps[-1].h}\n")

    for token_id, token in ctx.id_to_token.items():
        probability = probabilities[token_id]

        markers = []
        if token_id == prediction_id:
            markers.append("predicted")
        if token_id == ctx.target_token_id:
            markers.append("correct")
        marker = "".join(f"  <- {m}" for m in markers)

        print(
            f"  [{token:>5} ]: score = {scores[token_id]:+.3f}  |  "
            f"probability = {probability:.3f} ({probability:.0%}){marker}"
        )

    predicted = ctx.id_to_token[prediction_id]
    print(f"\nPredicted token:  '{predicted}'")
    print(f"Full prediction:  '{' '.join(ctx.input_tokens)} {predicted}'")
    print(f"\nCross-entropy loss: {loss:.3f}")

    takeaway(
        "The model is untrained, so its prediction is essentially a guess.",
        "A lower loss means more probability on the correct token.",
    )


@showcase.experiment
def training(ctx: Context) -> None:
    """Training with backpropagation

    Backpropagation through time (BPTT) works out how each weight affects the loss,
    and the update nudges every weight to lower it.
    Repeating this on the same example teaches the model to predict the target.
    """
    model = ctx.model
    target_id = ctx.target_token_id

    _, probabilities = model.predict(ctx.x_sequence)
    probability_before = probabilities[target_id]

    # losses[n] is the loss after n updates.
    losses = []
    for _ in range(TRAINING_STEPS):
        model.forward(ctx.x_sequence)
        losses.append(model.compute_loss(target_id))
        model.backward(target_id)

    prediction_id, probabilities = model.predict(ctx.x_sequence)
    losses.append(model.compute_loss(target_id))

    print("Loss during training:")
    for n in (0, 1, 10, 50, TRAINING_STEPS):
        print(f"  after {n:>3} updates: {losses[n]:.4f}")

    print(f"\nFirst update changed the loss by {losses[1] - losses[0]:+.4f}")

    predicted = ctx.id_to_token[prediction_id]
    print(f"Target probability:  {probability_before:.1%}  ->  {probabilities[target_id]:.1%}")
    print(f"Predicted token:     '{predicted}'  (correct: '{ctx.target_token}')")
    print(f"Full prediction:     '{' '.join(ctx.input_tokens)} {predicted}'")

    takeaway(
        "Each update shifts the weights to put more probability on the correct token.",
        "The model only sees one example, so it can memorize it.",
        "That is not the same as learning the language (see the training block).",
    )


# =====================================
# Runner
# =====================================

def main() -> None:
    ctx = build_context()
    show_setup(ctx)
    showcase.run(ctx)


if __name__ == "__main__":
    main()