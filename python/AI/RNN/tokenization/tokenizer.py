import string


# =====================================
# Tokenization
def character_tokenize(text):
    """creates tokens from characters, e.g. ['c','a','t']"""
    return list(text)


def word_tokenize(text):
    """Creates tokens from words, e.g.g ['the', 'cat']"""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


# =====================================
# Special tokens

def add_special_tokens(tokens):
    """Adds beginning and enf of sequence tokens (BOS and EOS)"""
    return ["<BOS>"] + tokens + ["<EOS"]


# =====================================
# Vocabulary

def create_vocabulary(tokens):
    """Creates a vocabulary from tokens, making encoding and decodingpossible"""
    vocab = ["<UNK>"] + sorted(set(tokens))

    # encoding
    token_to_id = {
        token: i
        for i, token in enumerate(vocab)
    }

    # deciding
    id_to_token = {
        i: token
        for token, i in token_to_id.items()
    }

    return vocab, token_to_id, id_to_token


# =====================================
# Encoding / decoding

def encode(tokens, token_to_id):
    """Encodes a set of given tokens, with unknown ones in mind"""
    unk_id = token_to_id["<UNK>"]
    return [token_to_id.get(token, unk_id) for token in tokens]


def decode(encoded, id_to_token):
    return [id_to_token[i] for i in encoded]


# =====================================
# Sequences

def create_sequences(encoded, sequence_length):
    inputs = []
    targets = []

    # Go through 4 characters at a time and use the next character as the target.
    # e.g.
    # 'hell' -> the next character in the text is 'o', therefore:
    # inputs = [h, e, l, l], targets = [o]
    for i in range(len(encoded) - sequence_length):
        input_ids = encoded[i:i + sequence_length]
        target_id = encoded[i + sequence_length]

        inputs.append(input_ids)
        targets.append(target_id)

    return inputs, targets
