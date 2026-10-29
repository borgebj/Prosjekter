import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentRNN(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 8,
            hidden_size: int = 16,
            output_size: int = 2,
    ):
        super().__init__()

        # embedding layer: tokens to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # main RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        # hidden to output
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (sequence_length,) -> (1, sequence_length)
        token_ids = token_ids.unsqueeze(0)

        # Token IDs -> learned word vectors
        embedded = self.embedding(token_ids)

        # process the sequence
        _, hidden = self.rnn(embedded)

        # classify using final hidden state
        logits = self.output(hidden[-1])

        return logits


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class NextRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        padding_idx=None,
        dropout=0.1,
        use_layer_norm=True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        # Regularization: randomly drop embedding activations during training.
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Normalize the final hidden state before prediction.
        self.layer_norm = (
            nn.LayerNorm(hidden_size)
            if use_layer_norm
            else nn.Identity()
        )

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, lengths=None):
        # input_ids: (batch_size, sequence_length)
        embedded = self.embedding(input_ids)

        # add regularization to embeddings
        embedded = self.dropout(embedded)

        if lengths is not None:
            packed = pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            _, hidden = self.rnn(packed)
            final_hidden = hidden[-1]
        else:
            outputs, _ = self.rnn(embedded)
            final_hidden = outputs[:, -1, :]

        # normalize before output layer
        final_hidden = self.layer_norm(final_hidden)

        # final linear layer
        logits = self.output(final_hidden)

        # Shape: (batch_size, vocab_size)
        return logits