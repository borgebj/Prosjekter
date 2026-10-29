from typing import List, Tuple

from AI.RNN.tokenization.tokenizer import (
    word_tokenize,
    create_vocabulary,
    encode,
)


# Labels: 0 = negative, 1 = positive

TRAIN_SAMPLES: List[Tuple[str, int]] = [
    # Positive
    ("i love this movie", 1),
    ("this movie is great", 1),
    ("i really like this", 1),
    ("what a wonderful day", 1),
    ("this is very good", 1),
    ("i enjoy this movie", 1),
    ("the food tastes amazing", 1),
    ("she is a great friend", 1),
    ("i had a fantastic time", 1),
    ("this book is excellent", 1),
    ("the experience was lovely", 1),
    ("i am happy with this", 1),
    ("the weather is beautiful", 1),
    ("we had a great evening", 1),
    ("this song sounds wonderful", 1),
    ("i love the service", 1),
    ("the game is fun", 1),
    ("that was a brilliant idea", 1),
    ("the staff were friendly", 1),
    ("everything worked perfectly", 1),
    ("i really love this book", 1),
    ("the movie was wonderful", 1),
    ("this game is really fun", 1),
    ("we enjoyed a lovely evening", 1),
    ("the food was very good", 1),
    ("she is a wonderful friend", 1),
    ("i am happy with the service", 1),
    ("the experience was fantastic", 1),
    ("that was a great idea", 1),
    ("everything worked very well", 1),
    ("i like this song", 1),
    ("the book is excellent", 1),
    ("we had a wonderful time", 1),
    ("the staff were great", 1),
    ("this is a brilliant game", 1),
    ("i enjoy the experience", 1),
    ("the weather was beautiful", 1),
    ("she had a fantastic evening", 1),
    ("the service was excellent", 1),
    ("this movie is very good", 1),
    ("i love the food", 1),
    ("the game was brilliant", 1),
    ("we are happy with this", 1),
    ("that was a lovely idea", 1),
    ("the song is wonderful", 1),
    ("i had a great time", 1),
    ("everything is perfectly fine", 1),
    ("this friend is amazing", 1),
    ("the evening was beautiful", 1),
    ("i really enjoy this", 1),
    ("pasta is my favorite food", 1),
    ("my life is great", 1),
    ("i had a great time", 1),
    ("this made me happy", 1),

    # Negative
    ("i hate this movie", 0),
    ("this movie is terrible", 0),
    ("i really dislike this", 0),
    ("what a horrible day", 0),
    ("this is very bad", 0),
    ("i regret watching this", 0),
    ("the food tastes awful", 0),
    ("he is a terrible friend", 0),
    ("i had a miserable time", 0),
    ("this book is boring", 0),
    ("the experience was unpleasant", 0),
    ("i am unhappy with this", 0),
    ("the weather is awful", 0),
    ("we had a terrible evening", 0),
    ("this song sounds horrible", 0),
    ("i hate the service", 0),
    ("the game is boring", 0),
    ("that was a stupid idea", 0),
    ("the staff were rude", 0),
    ("everything worked badly", 0),
    ("i really hate this book", 0),
    ("the movie was horrible", 0),
    ("this game is very boring", 0),
    ("we had an unpleasant evening", 0),
    ("the food was very bad", 0),
    ("he is a horrible friend", 0),
    ("i am unhappy with the service", 0),
    ("the experience was miserable", 0),
    ("that was a terrible idea", 0),
    ("everything worked very badly", 0),
    ("i dislike this song", 0),
    ("the book is terrible", 0),
    ("we had a horrible time", 0),
    ("the staff were unpleasant", 0),
    ("this is a stupid game", 0),
    ("i regret the experience", 0),
    ("the weather was awful", 0),
    ("he had a miserable evening", 0),
    ("the service was terrible", 0),
    ("this movie is very bad", 0),
    ("i hate the food", 0),
    ("the game was horrible", 0),
    ("we are unhappy with this", 0),
    ("that was a stupid game", 0),
    ("the song is horrible", 0),
    ("i had a terrible time", 0),
    ("everything worked badly today", 0),
    ("this friend is terrible", 0),
    ("the evening was unpleasant", 0),
    ("i really dislike this movie", 0),
    ("i really hate chicken", 0),
    ("my life is misreable", 0),
    ("i dont enjoy doing this", 0),
    ("i dont find this good", 0)
]


TEST_SAMPLES: List[Tuple[str, int]] = [
    # Positive
    ("i love the game", 1),
    ("this movie is wonderful", 1),
    ("the food is amazing", 1),
    ("we enjoy a great movie", 1),
    ("the staff were friendly", 1),
    ("i had a lovely evening", 1),
    ("this song is beautiful", 1),
    ("everything worked perfectly", 1),
    ("she is a brilliant friend", 1),
    ("the experience was excellent", 1),

    # Negative
    ("i hate the game", 0),
    ("this movie is horrible", 0),
    ("the food is awful", 0),
    ("we dislike this movie", 0),
    ("the staff were rude", 0),
    ("i had a miserable evening", 0),
    ("this song is terrible", 0),
    ("everything worked badly", 0),
    ("she is a terrible friend", 0),
    ("the experience was unpleasant", 0),
]


def prepare_data():
    """Tokenize data and build vocabulary from training samples only."""

    train_tokenized = [
        word_tokenize(sentence)
        for sentence, _ in TRAIN_SAMPLES
    ]

    test_tokenized = [
        word_tokenize(sentence)
        for sentence, _ in TEST_SAMPLES
    ]

    # Build vocabulary only from training data.
    vocabulary, token_to_id, id_to_token = create_vocabulary(
        [
            token
            for sentence in train_tokenized
            for token in sentence
        ]
    )

    # Unknown words in test data receive the <UNK> token ID.
    train_inputs = [
        encode(tokens, token_to_id)
        for tokens in train_tokenized
    ]

    test_inputs = [
        encode(tokens, token_to_id)
        for tokens in test_tokenized
    ]

    train_labels = [label for _, label in TRAIN_SAMPLES]
    test_labels = [label for _, label in TEST_SAMPLES]

    return (
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        vocabulary,
        token_to_id,
        id_to_token,
    )