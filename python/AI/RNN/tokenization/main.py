
from tokenizer import (
    character_tokenize,
    word_tokenize,
    BPETokenizer,
    create_vocabulary,
    encode,
    decode,
    create_sequences,
    add_special_tokens
)

num_experiments = 1


def print_header(text: str) -> None:
    global num_experiments
    print("\n\n" + "=" * 60)
    print(f"EXPERIMENT {num_experiments}: {text}")
    print("=" * 60)
    num_experiments += 1


corpus = [
    "The cat sat on the mat.",
    "The dog sat beside the cat.",
    "The cat was happy.",
]


def test_tokenizer(
    name: str,
    tokenizer,
    text: str,
    sequence_length: int
) -> None:
    print_header(f"{name.upper()} TOKENIZATION")

    tokens = tokenizer(text)
    tokens = add_special_tokens(tokens)
    vocab, token_to_id, id_to_token = create_vocabulary(tokens)
    encoded = encode(tokens, token_to_id)
    inputs, targets = create_sequences(encoded, sequence_length)

    print("\nTokens:", tokens)
    print("Vocabulary:", vocab)
    print("Encoded:", encoded)

    print("\nSequences:")
    for i, (input_ids, target_id) in enumerate(zip(inputs, targets)):
        if i > 20: print("..."); break

        input_tokens = decode(input_ids, id_to_token)
        target_token = decode([target_id], id_to_token)

        if name == "character":
            input_text = "".join(input_tokens)
            target_text = "".join(target_token)
        else:
            input_text = " ".join(input_tokens)
            target_text = " ".join(target_token)

        print(f"{input_text!r} -> {target_text!r}")

    print("\nStatistics:")
    print(f"Tokens: {len(tokens)}")
    print(f"Vocabulary: {len(vocab)}")
    print(f"Sequences: {len(inputs)}")


# Corpus
print("\nCorpus used:\n")
combined_text = "\n".join(corpus)
print(combined_text)

# Character-level
test_tokenizer(
    "character",
    character_tokenize,
    combined_text,
    sequence_length=5
)

# Word-level
test_tokenizer(
    "word",
    word_tokenize,
    combined_text,
    sequence_length=5
)


# =====================================
# BPE TOKENIZATION
print_header("BPE TOKENIZATION")

bpe = BPETokenizer()
merges = bpe.train(corpus, num_merges=20)

# Tokenize each sentence separately, then combine the tokens.
tokens = []
for sentence in corpus:
    tokens.extend(bpe.tokenize(sentence))

tokens = add_special_tokens(tokens)
vocab, token_to_id, id_to_token = create_vocabulary(tokens)
encoded = encode(tokens, token_to_id)
inputs, targets = create_sequences(encoded, sequence_length=5)

print("\nLearned merge rules:")
for left, right in merges:
    print(f"  {left!r} + {right!r} -> {left + right!r}")

print("\nTokens:", tokens)
print("Vocabulary:", vocab)
print("Encoded:", encoded)
print(
    f"\nStatistics: {len(tokens)} tokens, "
    f"{len(vocab)} vocabulary entries, "
    f"{len(inputs)} sequences"
)


# =====================================
# UNKNOWN TOKEN TEST
print_header("UNKNOWN TOKEN TEST")

training_tokens = add_special_tokens(word_tokenize(combined_text))
_, token_to_id, id_to_token = create_vocabulary(training_tokens)

test_tokens = word_tokenize("The bird sat outside")
encoded = encode(test_tokens, token_to_id)
decoded = decode(encoded, id_to_token)

print("\nTest tokens:", test_tokens)
print("Encoded:", encoded)
print("Decoded:", decoded)
