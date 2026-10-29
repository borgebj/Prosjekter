import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class NextTokenDataset(Dataset):
    def __init__(self, encoded_stories, max_context=16):
        self.examples = []

        for ids in encoded_stories:
            for i in range(1, len(ids)):
                start = max(0, i - max_context)
                input_ids = ids[start:i]
                target_id = ids[i]

                self.examples.append((input_ids, target_id))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_ids, target_id = self.examples[index]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_id, dtype=torch.long),
        )


def collate_batch(batch, pad_id):
    sequences, targets = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=pad_id,
    )

    targets = torch.stack(targets)

    return padded_sequences, targets, lengths


def create_dataloader(
    encoded_stories,
    batch_size=64,
    max_context=16,
    pad_id=0,
    shuffle=True,
):
    dataset = NextTokenDataset(
        encoded_stories,
        max_context=max_context,
    )

    def collate_fn(batch):
        return collate_batch(batch, pad_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )