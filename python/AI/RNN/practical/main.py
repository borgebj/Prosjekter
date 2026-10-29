from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from AI.RNN.showcase import Showcase, header, takeaway
from AI.RNN.tokenization.tokenizer import (
    BPETokenizer,
    create_vocabulary,
    tokenize_corpus,
)
from AI.RNN.practical.rnn import NextRNN
from AI.RNN.practical.dataloader import create_dataloader


# =====================================
# Configuration
# =====================================

SEED = 69

# Model
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
DROPOUT = 0.1
USE_LAYER_NORM = True
WEIGHT_DECAY = 0.0

# Tokenization
NUM_MERGES = 200
MAX_CONTEXT = 32

# Data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.txt"

CORPUS_FORMAT = "lines"  # "separator", "lines", or "single"
STORY_SEPARATOR = "<|STORY|>"
MAX_STORIES = None

# Train / validation / test split
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10

# Training
BATCH_SIZE = 64
EPOCHS = 15

# Text generation
GENERATED_TOKENS = 30
START_TEXT = "the dog"
GENERATION_STRATEGY = "greedy"  # "greedy" or "sample"

# Special tokens
PAD_TOKEN = "<PAD>"

showcase = Showcase()


# =====================================
# Reproducibility
# =====================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================
# Data preparation
# =====================================

def load_corpus():
    """Load documents from the configured corpus file."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Corpus file not found: {}".format(CORPUS_PATH)
        )

    text = CORPUS_PATH.read_text(encoding="utf-8-sig")

    if CORPUS_FORMAT == "separator":
        if not STORY_SEPARATOR:
            raise ValueError(
                "STORY_SEPARATOR must be set in separator mode."
            )
        documents = text.split(STORY_SEPARATOR)

    elif CORPUS_FORMAT == "lines":
        documents = text.splitlines()

    elif CORPUS_FORMAT == "single":
        documents = [text]

    else:
        raise ValueError(
            "Unknown CORPUS_FORMAT: {}".format(CORPUS_FORMAT)
        )

    documents = [
        document.strip()
        for document in documents
        if document.strip()
    ]

    if MAX_STORIES is not None:
        documents = documents[:MAX_STORIES]

    if len(documents) < 3:
        raise ValueError(
            "Need at least 3 documents for train/validation/test splits."
        )

    return documents


def encode_stories(stories, token_to_id):
    """Convert tokenized documents into token IDs."""
    unk_id = token_to_id["<UNK>"]

    return [
        [token_to_id.get(token, unk_id) for token in tokens]
        for tokens in stories
    ]


def split_documents(documents):
    """Split documents before training the tokenizer or model."""
    documents = list(documents)
    random.shuffle(documents)

    total = len(documents)
    train_end = int(total * TRAIN_FRACTION)
    validation_end = int(
        total * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train_documents = documents[:train_end]
    validation_documents = documents[train_end:validation_end]
    test_documents = documents[validation_end:]

    if not train_documents or not validation_documents or not test_documents:
        raise ValueError(
            "The corpus is too small for the configured data splits."
        )

    return train_documents, validation_documents, test_documents


def build_context():
    set_seed(SEED)

    documents = load_corpus()

    train_documents, validation_documents, test_documents = (
        split_documents(documents)
    )

    # Learn BPE merges using training documents only.
    bpe = BPETokenizer()
    bpe.train("\n".join(train_documents), num_merges=NUM_MERGES)

    train_tokens = tokenize_corpus(train_documents, bpe)
    validation_tokens = tokenize_corpus(validation_documents, bpe)
    test_tokens = tokenize_corpus(test_documents, bpe)

    flat_train_tokens = [
        token
        for document in train_tokens
        for token in document
    ]

    vocabulary, token_to_id, id_to_token = create_vocabulary(
        flat_train_tokens
    )

    # Reserve a dedicated ID for padding.
    if PAD_TOKEN in token_to_id:
        pad_id = token_to_id[PAD_TOKEN]
    else:
        pad_id = len(vocabulary)
        vocabulary.append(PAD_TOKEN)
        token_to_id[PAD_TOKEN] = pad_id
        id_to_token[pad_id] = PAD_TOKEN

    train_ids = encode_stories(train_tokens, token_to_id)
    validation_ids = encode_stories(validation_tokens, token_to_id)
    test_ids = encode_stories(test_tokens, token_to_id)

    train_loader = create_dataloader(
        train_ids,
        batch_size=BATCH_SIZE,
        max_context=MAX_CONTEXT,
        pad_id=pad_id,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        validation_ids,
        batch_size=BATCH_SIZE,
        max_context=MAX_CONTEXT,
        pad_id=pad_id,
        shuffle=False,
    )

    test_loader = create_dataloader(
        test_ids,
        batch_size=BATCH_SIZE,
        max_context=MAX_CONTEXT,
        pad_id=pad_id,
        shuffle=False,
    )

    model = NextRNN(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        padding_idx=pad_id,
        dropout=DROPOUT,
        use_layer_norm=USE_LAYER_NORM,
    )

    return {
        "bpe": bpe,
        "vocabulary": vocabulary,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "pad_id": pad_id,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "model": model,
    }


# =====================================
# Training and evaluation
# =====================================

def evaluate(model, data_loader, criterion):
    """Return average token loss and next-token accuracy."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            logits = model(inputs)

            batch_size, sequence_length, vocab_size = logits.shape

            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

            num_tokens = targets.numel()
            total_loss += loss.item() * num_tokens
            total_correct += (
                logits.argmax(dim=-1) == targets
            ).sum().item()
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("nan"), float("nan")

    return (
        total_loss / total_tokens,
        total_correct / total_tokens,
    )


def train(
    model,
    train_loader,
    validation_loader,
    epochs,
    lr,
    weight_decay=0.0,
):
    """Train the model and restore the best validation checkpoint."""
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_tokens = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            logits = model(inputs)
            batch_size, sequence_length, vocab_size = logits.shape

            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            num_tokens = targets.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        train_loss = total_loss / max(total_tokens, 1)

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
        )

        print(
            "Epoch {:>2}/{} | Train loss: {:.4f} "
            "| Validation loss: {:.4f} | Validation accuracy: {:.1%}"
            .format(
                epoch + 1,
                epochs,
                train_loss,
                validation_loss,
                validation_accuracy,
            )
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_validation_loss


@showcase.experiment
def next_token_demo(ctx):
    model = ctx["model"]
    train_loader = ctx["train_loader"]
    validation_loader = ctx["validation_loader"]
    test_loader = ctx["test_loader"]

    criterion = nn.CrossEntropyLoss()

    header("PYTORCH RNN: NEXT-TOKEN PREDICTION")

    print("Task: predict the next token in a text sequence.")
    print("Corpus: {}".format(CORPUS_PATH.name))
    print("Corpus format: {}".format(CORPUS_FORMAT))
    print("Vocabulary size: {}".format(len(ctx["vocabulary"])))
    print("Training batches: {}".format(len(train_loader)))
    print("Validation batches: {}".format(len(validation_loader)))
    print("Test batches: {}".format(len(test_loader)))
    print("Batch size: {}".format(BATCH_SIZE))
    print("Dropout: {}".format(DROPOUT))
    print("LayerNorm: {}".format(USE_LAYER_NORM))
    print()

    model, best_validation_loss = train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    ctx["model"] = model

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\nBest validation loss: {:.4f}".format(best_validation_loss))
    print("Test loss: {:.4f}".format(test_loss))
    print("Test next-token accuracy: {:.1%}".format(test_accuracy))

    takeaway(
        "The RNN predicts the next token from previous tokens.",
        "Validation loss helps compare model configurations.",
    )


# =====================================
# Text generation
# =====================================

def generate_text(ctx, start_text, num_tokens):
    model = ctx["model"]
    token_to_id = ctx["token_to_id"]
    id_to_token = ctx["id_to_token"]
    bpe = ctx["bpe"]
    pad_id = ctx["pad_id"]

    tokens = bpe.tokenize(start_text)

    if not tokens:
        raise ValueError("The starting text produced no tokens.")

    unk_id = token_to_id["<UNK>"]

    token_ids = [
        token_to_id.get(token, unk_id)
        for token in tokens
    ]

    model.eval()

    with torch.no_grad():
        for _ in range(num_tokens):
            context_ids = token_ids[-MAX_CONTEXT:]

            inputs = torch.tensor(
                [context_ids],
                dtype=torch.long,
            )

            # Predict the token following the final context token.
            logits = model(inputs)[0, -1].clone()

            # Padding is a batching utility, not a text token.
            logits[pad_id] = float("-inf")

            if GENERATION_STRATEGY == "greedy":
                next_id = int(logits.argmax().item())

            elif GENERATION_STRATEGY == "sample":
                probabilities = torch.softmax(logits, dim=0)
                next_id = int(
                    torch.multinomial(probabilities, 1).item()
                )

            else:
                raise ValueError(
                    "Unknown generation strategy: {}".format(
                        GENERATION_STRATEGY
                    )
                )

            token_ids.append(next_id)

    generated_tokens = [
        id_to_token.get(token_id, "<UNK>")
        for token_id in token_ids
    ]

    return "".join(generated_tokens).replace("▁", " ").strip()


@showcase.experiment
def generation_demo(ctx):
    header("GENERATE TEXT")

    print("Starting text: {}\n".format(START_TEXT))

    for sample_number in range(3):
        text = generate_text(
            ctx,
            start_text=START_TEXT,
            num_tokens=GENERATED_TOKENS,
        )

        print("Sample {}:".format(sample_number + 1))
        print(text)
        print()


# =====================================
# Interactive testing
# =====================================

@showcase.experiment
def interactive_demo(ctx):
    header("TRY YOUR OWN PROMPT")

    print("Enter a starting phrase, or type 'quit' to finish.")

    while True:
        prompt = input("\nPrompt: ").strip()

        if prompt.lower() in ("quit", "exit", "q"):
            break

        if not prompt:
            continue

        print(
            generate_text(
                ctx,
                start_text=prompt,
                num_tokens=GENERATED_TOKENS,
            )
        )


# =====================================
# Experiments
# =====================================

EXPERIMENTS = {
    "run": [
        next_token_demo,
        generation_demo,
        interactive_demo,
    ],
}


def main():
    ctx = build_context()

    for experiment in EXPERIMENTS["run"]:
        experiment(ctx)


if __name__ == "__main__":
    main()