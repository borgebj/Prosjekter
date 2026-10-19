"""
Block 2: Embeddings showcase.

Run this file to walk through the core ideas behind embeddings.
All experiments share one sentence, one vocabulary and one embedding
matrix (see Context), so the numbers can be followed from one
experiment to the next.
"""
from AI.RNN.showcase import Showcase, header, takeaway
from AI.RNN.tokenization.tokenizer import word_tokenize, create_vocabulary
from AI.RNN.embedding.embeddings import (
    one_hot_encode,
    create_embedding_matrix,
    positional_encoding,
)
from dataclasses import dataclass
import numpy as np

np.set_printoptions(
    precision=2,
    suppress=True,
    floatmode="fixed",
    sign="+",
)

EMBEDDING_DIM = 4
TEXT = "The cat sat on the mat."

showcase = Showcase()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator)


# =====================================
# Shared data (Block 1 reused)
# =====================================

@dataclass
class Context:
    """Data shared by every experiment."""
    tokens: list[str]                # ['the', 'cat', 'sat', 'on', 'the', 'mat', '.']
    vocab: list[str]                 # unique tokens
    token_to_id: dict[str, int]      # {'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4, '.': 5}
    token_ids: list[int]             # the sentence as IDs
    embedding_matrix: np.ndarray     # E: (vocab size, embedding dim)
    sequence: np.ndarray             # E[token_ids]: (sequence length, embedding dim)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_context() -> Context:
    """Block 1: text -> tokens -> IDs. Then create E and look up the sentence."""
    tokens = word_tokenize(TEXT)
    vocab, token_to_id, _ = create_vocabulary(tokens)
    token_ids = [token_to_id[t] for t in tokens]
    embedding_matrix = create_embedding_matrix(len(vocab), EMBEDDING_DIM)

    return Context(
        tokens=tokens,
        vocab=vocab,
        token_to_id=token_to_id,
        token_ids=token_ids,
        embedding_matrix=embedding_matrix,
        sequence=embedding_matrix[token_ids],
    )


def show_setup(ctx: Context) -> None:
    header("SETUP: TOKENIZATION AND VOCABULARY (Block 1)")
    print(f"Text:       {TEXT}")
    print(f"Tokens:     {ctx.tokens}")
    print(f"Vocabulary: {ctx.vocab}")
    print(f"Token IDs:  {ctx.token_ids}")


# =====================================
# Experiments
# =====================================

@showcase.experiment
def one_hot_to_embedding(ctx: Context) -> None:
    """From one-hot vectors to embeddings

    A one-hot vector has a single 1 at the token's ID.
    Multiplying it by the embedding matrix E selects one row of E: the token's embedding.
    Looking the row up directly gives the same result without building the one-hot vector.
    """
    one_hot = np.stack(
        [one_hot_encode(i, ctx.vocab_size) for i in ctx.token_ids]
    )

    for token, token_id, vector in zip(ctx.tokens, ctx.token_ids, one_hot):
        print(f"{token:>5} (ID {token_id}) -> {vector.astype(int)}")

    equal = np.allclose(one_hot @ ctx.embedding_matrix, ctx.sequence)
    print(f"\n(one-hot vectors) @ E  ==  E[token IDs]?  {equal}")

    takeaway(
        "One-hot vectors identify tokens but encode no relationships.",
        "An embedding layer is just this row lookup, done efficiently.",
    )


@showcase.experiment
def embedding_matrix(ctx: Context) -> None:
    """The embedding matrix

    E has one row per vocabulary token and one column per embedding dimension.
    Its values start out random, and training adjusts them.
    """
    E = ctx.embedding_matrix

    for token in ctx.vocab:
        token_id = ctx.token_to_id[token]
        print(f"{token:>5} (ID {token_id}) -> {E[token_id]}")

    print(f"\nShape of E:        {E.shape}  (vocab size, embedding dim)")
    print(f"Shape of sentence: {ctx.sequence.shape}  (sequence length, embedding dim)")

    # The same token always maps to the same row.
    positions = [i for i, t in enumerate(ctx.tokens) if t == "the"]
    identical = np.allclose(ctx.sequence[positions[0]], ctx.sequence[positions[1]])
    print(f"\n'the' appears at positions {positions}. Identical rows? {identical}")

    # Random vectors carry no meaning yet.
    similarity = cosine_similarity(
        E[ctx.token_to_id["cat"]],
        E[ctx.token_to_id["mat"]],
    )
    print(f"Cosine similarity (cat vs mat): {similarity:+.4f}")

    takeaway(
        "Repeated tokens share one row, wherever they appear.",
        "The similarity above is noise from random initialization.",
    )


@showcase.experiment
def positional_encoding_demo(ctx: Context) -> None:
    """Positional encoding (Transformer preview)

    Embeddings do not say where a token sits in the sentence.
    A sinusoidal position vector is added to each embedding to fix that.
    """
    for position, (token, embedding) in enumerate(zip(ctx.tokens, ctx.sequence)):
        pos_vector = positional_encoding(position, EMBEDDING_DIM)

        print(f"Position {position}: '{token}'")
        print(f"  {'Embedding:':<10}{embedding}")
        print(f"  {'Position:':<10}{pos_vector}")
        print(f"  {'Combined:':<10}{embedding + pos_vector}\n")

    takeaway(
        "'the' has the same embedding at both positions, ",
        "but a different combined vector. Order is now visible to the model.",
        "RNNs get order from their hidden state, so they don't need this.",
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