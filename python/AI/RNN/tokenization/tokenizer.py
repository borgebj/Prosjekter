import string
from collections import Counter
import re


# =====================================
# Tokenization
def character_tokenize(text: str) -> list[str]:
    """creates tokens from characters, e.g. ['c','a','t']"""
    return list(text)


def word_tokenize(text: str) -> list[str]:
    """Creates tokens from words, e.g.g ['the', 'cat']"""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


# =====================================
# BPE tokenization
class BPETokenizer:
    def __init__(self):
        self.merges = []

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        """Splits texts into words and punctuation."""
        return re.findall(r"\w+|[^\w\s]", text.lower())

    @staticmethod
    def get_pair_frequencies(tokens: list[str]) -> Counter[tuple[str, str]]:
        """Counts adjacent token pairs."""
        pairs = Counter()

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1

        return pairs

    def train(self, corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
        """Learns BPE merge rules from a corpus."""
        words = [
            [list(piece) for piece in self.pre_tokenize(text)]
            for text in corpus
        ]

        self.merges = []

        for _ in range(num_merges):
            pairs = Counter()

            # Count adjacent pairs across every piece in every sentence.
            for pieces in words:
                for tokens in pieces:
                    pairs.update(self.get_pair_frequencies(tokens))

            if not pairs:
                break

            # Select the most frequent pair; break ties lexicographically.
            highest = max(pairs.values())
            most_frequent = min(
                pair for pair, count in pairs.items()
                if count == highest
            )

            # Apply the merge to every piece in the corpus.
            new_words = []

            for pieces in words:
                new_pieces = []

                for tokens in pieces:
                    new_tokens = []
                    i = 0

                    while i < len(tokens):
                        if (
                                i + 1 < len(tokens)
                                and (tokens[i], tokens[i + 1]) == most_frequent
                        ):
                            new_tokens.append(tokens[i] + tokens[i + 1])
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1

                    new_pieces.append(new_tokens)

                new_words.append(new_pieces)

            self.merges.append(most_frequent)
            words = new_words

        return self.merges

    def tokenize(self, text: str) -> list[str]:
        """Applies learned BPE merge rules to new text."""
        result = []

        for piece in self.pre_tokenize(text):
            tokens = list(piece)

            for left, right in self.merges:
                new_tokens = []
                i = 0

                while i < len(tokens):
                    if (
                            i + 1 < len(tokens)
                            and (tokens[i], tokens[i + 1]) == (left, right)
                    ):
                        new_tokens.append(left + right)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                tokens = new_tokens

            result.extend(tokens)

        return result


# =====================================
# Special tokens

def add_special_tokens(tokens: list[str]) -> list[str]:
    """Adds beginning and enf of sequence tokens (BOS and EOS)"""
    return ["<BOS>"] + tokens + ["<EOS>"]


# =====================================
# Vocabulary

def create_vocabulary(tokens: list[str], include_unk: bool = True) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Creates a vocabulary from tokens, making encoding and decoding possible"""
    vocab = sorted(set(tokens))

    if include_unk:
        vocab = ["<UNK>"] + vocab

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

def encode(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    """Encodes a set of given tokens, with unknown ones in mind"""
    unk_id = token_to_id["<UNK>"]
    return [token_to_id.get(token, unk_id) for token in tokens]


def decode(encoded: list[int], id_to_token: dict[int, str]) -> list[str]:
    return [id_to_token[i] for i in encoded]


# =====================================
# Sequences

def create_sequences(encoded: list[int], sequence_length: int) -> tuple[list[list[int]], list[int]]:
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
