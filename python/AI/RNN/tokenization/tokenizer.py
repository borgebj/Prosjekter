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
        """Splits text into words and punctuation, marking word beginnings w/ '_'"""
        pieces = re.findall(r"\w+|[^\w\s]", text.lower())
        return ["▁" + piece if piece[0].isalnum() else piece for piece in pieces]

    @staticmethod
    def get_pair_frequencies(tokens: list[str]) -> Counter[tuple[str, str]]:
        """Counts adjacent token pairs."""
        return Counter(zip(tokens, tokens[1:])) if len(tokens) > 1 else Counter()

    @staticmethod
    def merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
        """Replaces every adjacent occurence of 'pair' with merged token"""
        merged, i = [], 0

        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
        """Learns BPE merge rules from a corpus."""
        words = [list(p) for text in corpus for p in self.pre_tokenize(text)]
        self.merges = []

        for _ in range(num_merges):
            pairs = Counter()
            for tokens in words:
                pairs.update(self.get_pair_frequencies(tokens))

            if not pairs:
                break

            # Most frequent pair - ties broken lexicographically
            best = min(pairs, key=lambda p: (-pairs[p], p))

            words = [self.merge_pair(tokens, best) for tokens in words]
            self.merges.append(best)

        return self.merges

    def tokenize(self, text: str) -> list[str]:
        """Applies learned BPE merge rules to new text."""
        result = []

        for piece in self.pre_tokenize(text):
            tokens = list(piece)

            for pair in self.merges:
                tokens = self.merge_pair(tokens, pair)
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


def tokenize_corpus(corpus: list[str], bpe: BPETokenizer):
    return [
        bpe.tokenize(sentence)
        for sentence in corpus
    ]
