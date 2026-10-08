import numpy as np


def cosine_similarity(a, b):
    """Calculates cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def print_similarities(embedding_matrix, token_to_id):
    comparisons = [
        ("cat", "dog"),
        ("car", "bus"),
        ("cat", "car"),
        ("dog", "bus")
    ]

    for word1, word2 in comparisons:
        a = embedding_matrix[token_to_id[word1]]
        b = embedding_matrix[token_to_id[word2]]

        similarity = cosine_similarity(a, b)

        print(f"{word1} - {word2}: {similarity:.4f}")


class EmbeddingTrainer:

    def __init__(
            self,
            embeddings,
            token_to_id,
            similar_pairs,
            different_pairs
    ):
        self.embeddings = embeddings
        self.token_to_id = token_to_id
        self.similar_pairs = similar_pairs
        self.different_pairs = different_pairs

    # =====================================
    # Distance
    # =====================================

    def distance(self, a, b):
        """Calculates the Euclidean distance between two embeddings."""
        return np.linalg.norm(a - b)

    # =====================================
    # Loss
    # =====================================

    def pair_loss(self, a, b, similar):
        """Calculates the loss for one pair of embeddings."""
        distance = self.distance(a, b)

        if similar:
            # Similar words should have a small distance.
            return distance

        # Different words should have a distance of at least 1.
        return max(0, 1 - distance)

    def calculate_loss(self):
        """Calculates the total loss for all word pairs."""
        loss = 0

        # Calculate loss for words that should be similar.
        for word1, word2 in self.similar_pairs:
            a = self.embeddings[self.token_to_id[word1]]
            b = self.embeddings[self.token_to_id[word2]]

            loss += self.pair_loss(a, b, similar=True)

        # Calculate loss for words that should be different.
        for word1, word2 in self.different_pairs:
            a = self.embeddings[self.token_to_id[word1]]
            b = self.embeddings[self.token_to_id[word2]]

            loss += self.pair_loss(a, b, similar=False)

        return loss

    # =====================================
    # Gradient
    # =====================================

    def pair_gradient(self, a, b, similar):
        """Calculates the gradients for one pair of embeddings."""

        # Difference between the two embeddings.
        difference = a - b

        # Distance between the two embeddings.
        distance = np.linalg.norm(difference)

        # Avoid division by zero if the vectors are identical.
        if distance == 0:
            return np.zeros_like(a), np.zeros_like(b)

        if similar:
            # Move similar embeddings towards each other.
            gradient_a = difference / distance
            gradient_b = -difference / distance

        else:
            # Move different embeddings apart,
            # but only if they are closer than the margin.
            if distance >= 1:
                return np.zeros_like(a), np.zeros_like(b)

            gradient_a = -difference / distance
            gradient_b = difference / distance

        return gradient_a, gradient_b

    # =====================================
    # Training
    # =====================================

    def train_step(self, learning_rate):
        """Performs one training step."""

        # Update similar pairs.
        for word1, word2 in self.similar_pairs:
            id1 = self.token_to_id[word1]
            id2 = self.token_to_id[word2]

            a = self.embeddings[id1]
            b = self.embeddings[id2]

            gradient_a, gradient_b = self.pair_gradient(
                a, b, similar=True
            )

            # Move embeddings in the opposite direction
            # of the gradient to reduce the loss.
            self.embeddings[id1] -= learning_rate * gradient_a
            self.embeddings[id2] -= learning_rate * gradient_b

        # Update different pairs.
        for word1, word2 in self.different_pairs:
            id1 = self.token_to_id[word1]
            id2 = self.token_to_id[word2]

            a = self.embeddings[id1]
            b = self.embeddings[id2]

            gradient_a, gradient_b = self.pair_gradient(
                a, b, similar=False
            )

            self.embeddings[id1] -= learning_rate * gradient_a
            self.embeddings[id2] -= learning_rate * gradient_b

    def train(self, learning_rate, epochs):
        """Trains the embeddings for a number of epochs."""

        # repeat 'epochs' times
        for epoch in range(epochs):
            loss = self.calculate_loss()

            if epoch % (epochs // 5) == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")

            self.train_step(learning_rate)
