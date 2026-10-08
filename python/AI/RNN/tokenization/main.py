from tokenizer import (
    character_tokenize,
    word_tokenize,
    create_vocabulary,
    encode,
    decode,
    create_sequences,
    add_special_tokens
)


text = """\
The cat sat on the mat.
The cat was happy.
The dog sat beside the cat.
"""


def test_tokenizer(name, tokenizer, text, sequence_length):
    print("\n" + "=" * 50)
    print(f"{name.upper()} TOKENIZATION")
    print("=" * 50)

    tokens = tokenizer(text)
    tokens = add_special_tokens(tokens)
    vocab, token_to_id, id_to_token = create_vocabulary(tokens)
    encoded = encode(tokens, token_to_id)
    inputs, targets = create_sequences(encoded, sequence_length)

    print("\nTokens:", tokens)
    print("Vocabulary:", vocab)
    print("Encoded:", encoded)

    print("\nSequences:")
    for input_ids, target_id in zip(inputs, targets):
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


# Character-level
test_tokenizer(
    "character",
    character_tokenize,
    text,
    sequence_length=5
)

# Word-level
test_tokenizer(
    "word",
    word_tokenize,
    text,
    sequence_length=5
)


# =====================================
# Unknown token test

print("\n" + "=" * 50)
print("UNKNOWN TOKEN TEST")
print("=" * 50)

tokens = word_tokenize(text)
tokens = add_special_tokens(tokens)
vocab, token_to_id, id_to_token = create_vocabulary(tokens)

test_tokens = word_tokenize("The bird sat outside")

encoded = encode(test_tokens, token_to_id)
decoded = decode(encoded, id_to_token)

print("\nTest tokens:", test_tokens)
print("Encoded:", encoded)
print("Decoded:", decoded)