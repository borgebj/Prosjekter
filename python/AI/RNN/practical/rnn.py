import torch
import torch.nn as nn


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

        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.layer_norm = (
            nn.LayerNorm(hidden_size)
            if use_layer_norm
            else nn.Identity()
        )

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch_size, sequence_length)
        embedded = self.embedding(input_ids)

        # (B, T, embedding_dim)
        embedded = self.dropout(embedded)

        # (B, T, hidden_size)
        outputs, _ = self.rnn(embedded)

        # Normalize each timestep.
        outputs = self.layer_norm(outputs)

        # (B, T, vocab_size)
        logits = self.output(outputs)

        return logits