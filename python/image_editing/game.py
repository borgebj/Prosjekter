import random

from filters import images
from filters import image_io
from filters import utility


def run():
    image_directory = "filters/../images"

    # Pick a random image
    filename = utility.random_image(image_directory)
    file = filename.rsplit(".", 1)[0]

    # Read image
    filename = f"{image_directory}/{filename}"
    img = image_io.read_image(filename)

    # Available filters
    available_filters = [
        ("greyscale", "numpy", {}),
        ("pixelator", "numpy", {"blocksize": 40}),
        ("ascii", "numpy", {"scale": 2}),
        ("sepia", "numpy", {}),
    ]

    # Randomly choose how many filters to use
    number_of_filters = random.randint(1, len(available_filters))

    # Choose filters and put them in random order
    filters = random.sample(available_filters, number_of_filters)

    # Run filters
    for filter_name, implementation, filter_args in filters:
        filter_fn = images.get_filter(filter_name, implementation)

        img, elapsed = utility.time_function(
            filter_fn,
            img,
            **filter_args
        )

        print(
            f'{"=" * 30}\n'
            f'{"Filter:":<20}{filter_name}\n'
            f'{"Implementation:":<20}{implementation}\n'
            f'{"Time:":<20}{elapsed:.4f}s\n'
            f'{"=" * 30}'
        )

    print(f'{"Image:":<20}{file}')

    image_io.display(img)


if __name__ == "__main__":
    run()