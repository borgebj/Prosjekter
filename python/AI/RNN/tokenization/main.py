"""
Block 1: Tokenization showcase.

Run this file to compare the main ways of turning text into tokens.
All experiments share one corpus, and each tokenizer (character, word, BPE)
is set up once on it (see Context), so the results can be compared directly.
"""
from AI.RNN.showcase import Showcase, style, header, takeaway
from AI.RNN.tokenization.tokenizer import (
    character_tokenize,
    word_tokenize,
    BPETokenizer,
    create_vocabulary,
    encode,
    decode,
    create_sequences,
    add_special_tokens,
)
from collections.abc import Callable
from dataclasses import dataclass

NUM_MERGES = 25
SEQUENCE_LENGTH = 5
MAX_SEQUENCES_SHOWN = 8

CORPUS = [
    "The cat was happy.",
    "The cats were unhappy.",
    "The dog was happier.",
    "The dogs were unhappier.",
    "The cat was playing.",
    "The cats were played with.",
]
EXAMPLE = "The dogs were unhappier."    # a sentence from the corpus
UNSEEN = "The bird was happy."          # 'bird' is not in the corpus

showcase = Showcase()


# =====================================
# Shared data
# =====================================

@dataclass
class Scheme:
    """One tokenizer applied to the whole corpus."""
    name: str
    tokenize: Callable[[str], list[str]]
    tokens: list[str]               # corpus as one token stream, with special tokens
    vocab: list[str]
    token_to_id: dict
    id_to_token: dict


@dataclass
class Context:
    """Data shared by every experiment."""
    corpus: list[str]
    schemes: dict[str, Scheme]      # 'character', 'word', 'BPE'
    bpe: BPETokenizer               # kept so experiments can show its merge rules


def build_scheme(name: str, tokenize: Callable[[str], list[str]]) -> Scheme:
    """Tokenizes each sentence, joins them, and builds the vocabulary."""
    tokens = [t for sentence in CORPUS for t in tokenize(sentence)]
    tokens = add_special_tokens(tokens)
    vocab, token_to_id, id_to_token = create_vocabulary(tokens)

    return Scheme(name, tokenize, tokens, vocab, token_to_id, id_to_token)


def build_context() -> Context:
    bpe = BPETokenizer()
    bpe.train(CORPUS, NUM_MERGES)

    schemes = {
        "character": build_scheme("character", character_tokenize),
        "word": build_scheme("word", word_tokenize),
        "BPE": build_scheme("BPE", bpe.tokenize),
    }
    return Context(CORPUS, schemes, bpe)


def show_setup(ctx: Context) -> None:
    header("SETUP: CORPUS")
    for sentence in ctx.corpus:
        print(sentence)


# =====================================
# Experiments
# =====================================

@showcase.experiment
def three_ways_to_split(ctx: Context) -> None:
    """Three ways to split text

    Character-level splits into single characters, word-level into words,
    and BPE into subwords by repeatedly merging the most frequent pair of neighbouring tokens.
    The same sentence and corpus are used for all three.
    """
    print(f"Sentence: {EXAMPLE!r}\n")
    for name, scheme in ctx.schemes.items():
        print(style(name, "1"))
        print(f"  {scheme.tokenize(EXAMPLE)}\n")

    print(f"\n{'Whole corpus':<14}{'tokens':>8}{'vocab':>8}")
    for name, scheme in ctx.schemes.items():
        print(f"{name:<14}{len(scheme.tokens):>8}{len(scheme.vocab):>8}")

    merges = ctx.bpe.merges
    print(f"\nFirst merge rules learned by BPE ({len(merges)} in total):")
    for left, right in merges[:10]:
        print(f"  {left!r} + {right!r} \t->\t {left + right!r}")

    takeaway(
        "Token counts: character-level is longest, word-level shortest, and BPE lands in between.",
        "On a corpus this small the vocabulary sizes are not representative.",
        "Word-level vocabularies are the ones that blow up on real text.",
    )


@showcase.experiment
def unseen_words(ctx: Context) -> None:
    """Unseen words

    A tokenizer only knows what it saw while building its vocabulary.
    Here each one encodes a sentence containing a word that is not in the corpus.
    Anything missing from the vocabulary becomes an unknown token.
    """
    print(f"Sentence: {UNSEEN!r}")

    for name, scheme in ctx.schemes.items():
        tokens = scheme.tokenize(UNSEEN)
        ids = encode(tokens, scheme.token_to_id)
        decoded = decode(ids, scheme.id_to_token)

        print(f"\n{style(name, '1')}")
        print(f"  {'Tokens:':<10}{tokens}")
        print(f"  {'Encoded:':<10}{ids}")
        print(f"  {'Decoded:':<10}{decoded}")

    takeaway(
        "Word-level loses the entire unseen word.",
        "Subword and character tokenizers lose at most the pieces they have never seen.",
    )


@showcase.experiment
def training_sequences(ctx: Context) -> None:
    """From tokens to training sequences

    Tokens are encoded as IDs, then cut into sliding windows.
    Each window of tokens (the input) is paired with the token that follows it (the target).
    This is the training data for a sequence model.
    """
    scheme = ctx.schemes["word"]
    encoded = encode(scheme.tokens, scheme.token_to_id)
    inputs, targets = create_sequences(encoded, SEQUENCE_LENGTH)

    print(f"Word-level, sequence length {SEQUENCE_LENGTH}\n")
    print(f"Encoded corpus: {encoded}\n")

    for i, (input_ids, target_id) in enumerate(zip(inputs, targets)):
        if i == MAX_SEQUENCES_SHOWN:
            print("...")
            break

        input_text = " ".join(decode(input_ids, scheme.id_to_token))
        target_text = " ".join(decode([target_id], scheme.id_to_token))
        print(f"{input_text!r:<45} -> {target_text!r}")

    print(f"\n{len(encoded)} tokens produce {len(inputs)} sequences.")

    takeaway(
        "The model learns to predict the next token from the previous ones.",
        "The same steps work for character-level and BPE tokens.",
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