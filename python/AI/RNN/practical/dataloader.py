import torch
from torch.utils.data import IterableDataset, DataLoader


class NextTokenDataset(IterableDataset):
    def __init__(
        self,
        encoded_stories,
        context_length=16,
        samples_per_epoch=None,
    ):
        self.encoded_stories = [
            torch.tensor(ids, dtype=torch.long)
            for ids in encoded_stories
            if len(ids) > context_length
        ]
        self.context_length = context_length
        self.samples_per_epoch = samples_per_epoch

        if not self.encoded_stories:
            raise ValueError(
                "No stories are longer than context_length."
            )

    def __iter__(self):
        count = self.samples_per_epoch

        if count is None:
            count = sum(
                len(story) - self.context_length
                for story in self.encoded_stories
            )

        for _ in range(count):

            # Select a random story.
            story_index = torch.randint(
                len(self.encoded_stories), (1,)
            ).item()

            tokens = self.encoded_stories[story_index]

            # Select a random starting position.
            start = torch.randint(
                len(tokens) - self.context_length,
                (1,),
            ).item()

            # creates examples:
            # 2 batch size, 4 context length
            # x = [ [2, 3, 4, 5],
            #       [7, 8, 9, 10] ]
            #
            # y = [ [3, 4, 5, 6],
            #       [8, 9, 10, 1] ]

            x = tokens[start:start + self.context_length]
            y = tokens[start + 1:start + self.context_length + 1]

            yield x, y

    def __len__(self):
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch

        return sum(
            len(story) - self.context_length
            for story in self.encoded_stories
        )


def create_dataloader(
    encoded_stories,
    batch_size=64,
    max_context=16,
    pad_id=0,
    shuffle=True,
    samples_per_epoch=None,
):
    dataset = NextTokenDataset(
        encoded_stories=encoded_stories,
        context_length=max_context,
        samples_per_epoch=samples_per_epoch,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
    )