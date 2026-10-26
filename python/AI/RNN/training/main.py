"""
Block 4: Training showcase.

Set MODE below, then run this file:

    train -> build a new model, train it on the training corpus, evaluate it, save it
    test  -> load the saved model and measure it on sentences it has never seen
    use   -> load the saved model and generate text

The tokenizer (Block 1), embeddings (Block 2) and RNN (Block 3) are reused,
and all experiments share one dataset and one model (see Context).

Run "train" first: "test" and "use" need the saved model. The tokenizer,
vocabulary and embeddings are rebuilt from SEED and the corpus, so keep those
(and the hyperparameters) unchanged between training and testing/using.
"""
from AI.RNN.showcase import Showcase, header, takeaway, bolden
from AI.RNN.model.rnn import RNN
from AI.RNN.tokenization.tokenizer import (
    create_vocabulary,
    BPETokenizer,
    tokenize_corpus,
)
from AI.RNN.embedding.embeddings import create_embedding_matrix
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import random
import re


# =====================================
# Settings
# =====================================

# Run mode
MODE = "train"  # "train", "test" or "use"

# Model hyperparameters
SEED = 100
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
LEARNING_RATE = 0.005

# Tokenizer hyperparameters
NUM_MERGES = 200
MAX_CONTEXT = 16

# Training hyperparameters
EPOCHS = 50
LOG_EVERY = (EPOCHS // 8) or 1

PATIENCE = 5
MIN_DELTA = 0.001

# Dataset split
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.1
# Remaining 20% is for testing.

# Output settings
GENERATED_TOKENS = 16
SAMPLES = 3
EXAMPLES_SHOWN = 5

# =====================================
# Dataset configurations
# =====================================

CONFIGS = {
    "tinystories": {
        "CORPUS_PATH": Path(__file__).with_name("tinystories.txt"),
        "START_TEXT": "Once upon a time",
        "MAX_ROWS": 200,
        "PREVIEW_LIMIT": 30,
    },
    "corpus": {
        "CORPUS_PATH": Path(__file__).with_name("corpus.txt"),
        "START_TEXT": "The dog",
        "MAX_ROWS": 1000,
        "PREVIEW_LIMIT": 30,
    },
}

# Select dataset
# ACTIVE_CONFIG = "corpus"
ACTIVE_CONFIG = "tinystories"


# =====================================
# Paths and active dataset
# =====================================

CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")

_config = CONFIGS[ACTIVE_CONFIG]

CORPUS_PATH = _config["CORPUS_PATH"]
START_TEXT = _config["START_TEXT"]
MAX_ROWS = _config["MAX_ROWS"]
PREVIEW_LIMIT = _config["PREVIEW_LIMIT"]

MODEL_NAME = f"{ACTIVE_CONFIG}_s{SEED}"

MODEL_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}.npz"
EMBEDDING_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_emb.npy"

showcase = Showcase()


# =====================================
# Shared data (Blocks 1, 2 and 3 reused)
# =====================================

@dataclass
class Context:
    """Data shared by every experiment."""
    train_corpus: list[str]
    validation_corpus: list[str]
    test_corpus: list[str]
    bpe: BPETokenizer
    train_tokenized: list[list[str]]  # training sentences as tokens
    vocab: list[str]  # built from training tokens only
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    unk_id: int  # unseen tokens map to this ID
    train_ids: list[list[int]]  # training sentences as IDs
    validation_ids: list[list[int]]
    test_ids: list[list[int]]
    embedding_matrix: np.ndarray  # E: (vocab size, embedding dim), trainable
    model: RNN  # model changed by the training experiment
    train_examples: list  # [(input ID, target ID), ...]
    validation_examples: list
    test_examples: list


def load_corpus() -> list[str]:
    text = CORPUS_PATH.read_text(encoding="utf-8-sig")

    if ACTIVE_CONFIG == "tinystories":
        entries = text.split("<|STORY|>")
    else:
        entries = text.splitlines()

    corpus = [
        re.sub(r"\s+", " ", entry).strip()
        for entry in entries
        if entry.strip()
    ]

    if MAX_ROWS is not None:
        corpus = corpus[:MAX_ROWS]

    return corpus


def save_embeddings(embedding_matrix: np.ndarray) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_PATH, embedding_matrix)


def encode_sentences(sentences, token_to_id, unk_id) -> list[list[int]]:
    """Converts tokens to IDs, using the unknown ID for unseen tokens."""
    return [
        [token_to_id.get(token, unk_id) for token in sentence]
        for sentence in sentences
    ]


def make_examples(id_sequences):
    """Create next-token examples using token IDs."""
    return [
        (ids[max(0, i - MAX_CONTEXT):i], ids[i])
        for ids in id_sequences
        for i in range(1, len(ids))
    ]


def evaluate(ctx: Context, examples: list) -> tuple[float, float]:
    """Returns average loss and accuracy without updating parameters."""
    if not examples:
        return float("nan"), float("nan")

    total_loss, correct = 0.0, 0
    for input_ids, target_id in examples:
        x_sequence = ctx.embedding_matrix[input_ids]
        prediction_id, _ = ctx.model.predict(x_sequence)
        total_loss += ctx.model.compute_loss(target_id)
        correct += prediction_id == target_id

    return total_loss / len(examples), correct / len(examples)


def generate(ctx: Context, start_text: str, num_tokens: int, rng=None) -> str:
    """Greedy generation, or sampling from the probabilities if rng is given."""
    tokens = ctx.bpe.tokenize(start_text)
    ids = [ctx.token_to_id.get(token, ctx.unk_id) for token in tokens]

    for _ in range(num_tokens):
        context_ids = ids[-MAX_CONTEXT:]
        x_sequence = ctx.embedding_matrix[context_ids]

        prediction_id, probabilities = ctx.model.predict(x_sequence)

        if rng is not None:
            p = np.asarray(probabilities, dtype=np.float64)
            prediction_id = rng.choice(len(p), p=p / p.sum())
        ids.append(int(prediction_id))

    # Reconstruct word boundaries from the ▁ marker.
    return "".join(ctx.id_to_token[i] for i in ids).replace("▁", " ").strip()


def build_context(new_model: bool) -> Context:
    """Builds the shared data. new_model=True starts from random weights,
    otherwise the saved model is loaded."""
    # Make experiments reproducible.
    random.seed(SEED)
    np.random.seed(SEED)

    # Load and split the corpus.
    corpus = load_corpus()
    random.shuffle(corpus)

    train_end = int(len(corpus) * TRAIN_SPLIT)
    val_end = int(len(corpus) * (TRAIN_SPLIT + VALIDATION_SPLIT))

    train_corpus = corpus[:train_end]
    validation_corpus = corpus[train_end:val_end]
    test_corpus = corpus[val_end:]

    # Block 1: tokenizer and vocabulary come from the training data only,
    # so the test sentences stay truly unseen.
    bpe = BPETokenizer()
    bpe.train(train_corpus, num_merges=NUM_MERGES)

    train_tokenized = tokenize_corpus(train_corpus, bpe)
    validation_tokenized = tokenize_corpus(validation_corpus, bpe)
    test_tokenized = tokenize_corpus(test_corpus, bpe)

    train_tokens = [token for sentence in train_tokenized for token in sentence]
    vocab, token_to_id, id_to_token = create_vocabulary(train_tokens)
    unk_id = token_to_id["<UNK>"]

    train_ids = encode_sentences(train_tokenized, token_to_id, unk_id)
    validation_ids = encode_sentences(validation_tokenized, token_to_id, unk_id)
    test_ids = encode_sentences(test_tokenized, token_to_id, unk_id)

    # Block 2: embeddings
    if new_model:
        embedding_matrix = np.asarray(
            create_embedding_matrix(len(vocab), EMBEDDING_DIM),
            dtype=np.float64,
        )
    else:
        embedding_matrix = np.load(EMBEDDING_PATH, allow_pickle=False)
        if embedding_matrix.shape != (len(vocab), EMBEDDING_DIM):
            raise ValueError(
                "Saved embedding matrix does not match the current vocabulary "
                "or embedding dimension."
            )

    # Block 3: model
    if new_model:
        model = RNN(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            output_size=len(vocab),
            lr=LEARNING_RATE,
        )
    else:
        model = RNN.load_model(name=MODEL_NAME)

    train_examples = make_examples(train_ids)
    validation_examples = make_examples(validation_ids)
    test_examples = make_examples(test_ids)

    if not train_examples:
        raise ValueError(
            "No training examples created. "
            "Check that the training corpus contains sentences with at least two tokens."
        )

    return Context(
        train_corpus, validation_corpus, test_corpus, bpe, train_tokenized, vocab, token_to_id, id_to_token, unk_id,
        train_ids, validation_ids, test_ids, embedding_matrix, model, train_examples, validation_examples,
        test_examples,
    )


def show_setup(ctx: Context) -> None:
    header("SETUP: DATA, TOKENIZER AND MODEL (Blocks 1-3)")
    print(f"Mode: '{MODE}'  |  Dataset: '{ACTIVE_CONFIG}'\n")

    print("Sample training sentences:")
    for sentence in ctx.train_corpus[:5]:
        print(f"  {sentence}")

    print(f"\n{bolden('Training sentences')}: {len(ctx.train_corpus)}")
    print(f"{bolden('Test sentences')}:       {len(ctx.test_corpus)}")
    print(f"{bolden('Validation sentences')}: {len(ctx.validation_corpus)}")
    print(f"{bolden('BPE merge rules')}:      {len(ctx.bpe.merges)}")
    print(f"{bolden('Vocabulary size')}:      {len(ctx.vocab)}")
    print(f"{bolden('Embedding dim')}:        {EMBEDDING_DIM}")
    print(f"{bolden('Hidden size')}:          {HIDDEN_SIZE}")
    print(f"{bolden('Learning rate')}:        {LEARNING_RATE}")
    print(f"\n{bolden('Vocabulary')} (first 15): {ctx.vocab[:15]}{' ...' if len(ctx.vocab) > 15 else ''}")


# =====================================
# Experiments
# =====================================

@showcase.experiment
def training_examples(ctx: Context) -> None:
    """From sentences to training examples

    -> Every prefix of a sentence is an input, and the token that follows it is the target.
    -> The input tokens are looked up in the embedding matrix, and the model learns to predict the target from them.
    """
    tokens, ids = ctx.train_tokenized[0], ctx.train_ids[0]

    print(f"\n{bolden('Sentence')}:\n\t  '{ctx.train_corpus[0]}'")
    print(f"\n{bolden('Tokens')}:\n\t   [{'  '.join(tokens[:PREVIEW_LIMIT])}]")
    print(f"\n{bolden('IDs')}:\n\t   [{'  '.join(str(id) for id in ids[:PREVIEW_LIMIT])}]\n")

    last = min(len(tokens), EXAMPLES_SHOWN + 1)
    for i in range(1, last):
        print(f"  {' '.join(tokens[:i])!r:<35} -> {tokens[i]!r}")
    if last < len(tokens):
        print("  ...")

    print(f"\n{len(ctx.train_corpus)} training sentences   ->  {len(ctx.train_examples)} examples")
    print(f"{len(ctx.test_corpus)} test sentences       ->  {len(ctx.test_examples)} examples")

    takeaway(
        "The first token of a sentence is never a target, because nothing comes before it.",
        "The test sentences are held back and never used to update the model.",
    )


@showcase.experiment
def training(ctx: Context) -> None:
    """Training

        -> One epoch is one pass over all training examples, in a new random order.
        -> For each example the model predicts the next token, measures the loss, and updates its weights.
        -> Validation loss is used for early stopping and checkpoint selection.
        """
    print("\n", bolden(("=" * 15) + "STARTING TRAINING" + ("=" * 15)))

    model = ctx.model
    examples = list(ctx.train_examples)

    best_validation_loss = float("inf")
    epochs_no_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(examples)
        total_loss = 0.0

        # forward + backward
        for input_ids, target_id in examples:

            # 1. look up current embeddings
            x_sequence = ctx.embedding_matrix[input_ids]

            # 2. forward + loss
            model.forward(x_sequence)
            total_loss += model.compute_loss(target_id)  # track loss

            # 3. calculate gradients and update RNN (backward)
            dx_sequence = model.backward(target_id)

            # 4. update embeddings
            embedding_gradients = np.zeros_like(ctx.embedding_matrix)

            for token_id, dx in zip(input_ids, dx_sequence):
                embedding_gradients[token_id] += dx

            ctx.embedding_matrix -= model.lr * embedding_gradients

        train_loss = total_loss / len(examples)
        validation_loss, _ = evaluate(ctx, ctx.validation_examples)

        if validation_loss < best_validation_loss - MIN_DELTA:
            best_validation_loss = validation_loss
            epochs_no_improvement = 0

            model.save_model(name=MODEL_NAME, silent=True)
            save_embeddings(ctx.embedding_matrix)
        else:
            epochs_no_improvement += 1

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(
                f"{bolden('Epoch')} {epoch:3d}/{EPOCHS}"
                f"  |  {bolden('train')} loss: {train_loss:.4f}"
                f"  |  {bolden('validation')} loss: {validation_loss:.4f}"
            )

        if epochs_no_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}.\n"
                f"Best validation loss: {best_validation_loss:.4f}"
            )
            break

    ctx.model = RNN.load_model(name=MODEL_NAME)
    ctx.embedding_matrix = np.load(EMBEDDING_PATH, allow_pickle=False)

    takeaway(
        "Falling train loss means the model fits the training data.",
        "If validation loss stops falling or rises while train loss keeps falling, the model may be overfitting.",
        "Both the RNN parameters and the embedding matrix are updated during training.",
    )


@showcase.experiment
def evaluation(ctx: Context) -> None:
    """Evaluation on unseen sentences

    -> The trained model is compared with random guessing, which has a loss of ln(V) for a vocabulary of V tokens.
    -> Perplexity is exp(loss): roughly how many tokens the model is choosing between at each step.
    """
    vocab_size = len(ctx.vocab)
    train_loss, train_accuracy = evaluate(ctx, ctx.train_examples)
    test_loss, test_accuracy = evaluate(ctx, ctx.test_examples)

    rows = [
        ("random guess", np.log(vocab_size), 1 / vocab_size),
        ("train", train_loss, train_accuracy),
        ("test", test_loss, test_accuracy),
    ]

    print(f"{'':<14}{'loss':>8}{'perplexity':>12}{'accuracy':>10}")
    for name, loss, accuracy in rows:
        print(f"{name:<14}{loss:>8.3f}{np.exp(loss):>12.1f}{accuracy:>10.1%}")

    test_tokens = sum(len(sentence) for sentence in ctx.test_ids)
    unknown = sum(i == ctx.unk_id for sentence in ctx.test_ids for i in sentence)
    print(f"\nTest tokens mapped to <UNK>: {unknown}/{test_tokens} ({unknown / max(test_tokens, 1):.1%})")

    takeaway(
        "A test loss clearly below the random baseline means the model learned patterns that transfer.",
        "A large gap between train and test loss means it memorized the training sentences.",
        "Accuracy is how often the single most likely token was the right one.",
    )


@showcase.experiment
def text_generation(ctx: Context) -> None:
    """Text generation

    -> The model writes one token at a time: it predicts a token, appends it to the text, and predicts again.
    -> Greedy decoding always takes the most likely token.
    -> Sampling draws from the predicted probabilities instead, which gives more variety.
    """
    rng = np.random.default_rng(SEED)

    print(f"{bolden('Start text')}: '{START_TEXT}'\n")

    print(f"\t{bolden('greedy')}:     {generate(ctx, START_TEXT, GENERATED_TOKENS)}")
    for n in range(1, SAMPLES + 1):
        sampled = generate(ctx, START_TEXT, GENERATED_TOKENS, rng)
        print(f"\t{bolden(f'sample {n}')}:   {sampled}")

    takeaway(
        "Greedy decoding is deterministic and tends to fall into loops.",
        "Sampling varies from run to run. With little data and a tiny model, expect rough text either way.",
    )


# TODO: trainable embeddings
# TODO: start/end markers and better generation


# =====================================
# Runner
# =====================================

# Which experiments each mode runs, in order.
EXPERIMENTS = {
    "train": [training_examples, training, evaluation],
    "test": [evaluation],
    "use": [text_generation],
}


def main():
    ctx = build_context(new_model=(MODE == "train"))
    show_setup(ctx)

    for experiment in EXPERIMENTS[MODE]:
        experiment(ctx)


if __name__ == "__main__":
    main()
