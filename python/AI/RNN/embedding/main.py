from AI.RNN.tokenization.tokenizer import (
    word_tokenize,
    create_vocabulary
)

from embeddings import one_hot_encode, dot_product

text = "The cat sat on the mat."

tokens = word_tokenize(text)
vocab, token_to_id, id_to_token = create_vocabulary(tokens)
vocab_size = len(vocab)

print(f"Tokens: {tokens}")
print(f"Vocabulary: {vocab}")
print(f"Token to ID: {token_to_id}\n")

# onehot encodes each token in vocabulary
one_hot_matrix = []

for token in vocab:
    token_id = token_to_id[token]
    vector = one_hot_encode(token_id, vocab_size)
    one_hot_matrix.append(vector)
    print(f"{token} -> {vector}")

print()
[print(x) for x in one_hot_matrix]

cat = one_hot_encode(token_to_id["cat"], vocab_size)
mat = one_hot_encode(token_to_id["mat"], vocab_size)
sat = one_hot_encode(token_to_id["sat"], vocab_size)

print("\n" + "="*10)
print("cat · mat:", dot_product(cat, mat))
print("mat · sat:", dot_product(mat, sat))
print("mat · mat:", dot_product(mat, mat))
