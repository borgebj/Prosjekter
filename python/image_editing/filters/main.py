import images
import image_io
from python.image_editing.filters import utility


def main():
    file = "mexico"
    filename = f"../images/{file}.jpg"
    filter_name = "blur"  # "pixelator", "ascii", "blur"
    implementation = "numpy"  # "python", "numpy"

    img = image_io.read_image(filename)

    # mostly just noise, still interesting
    # img = image_io.random_image(1920, 1200)

    # scaling
    img = utility.rescale(img, scale=1) # higher -> smaller resolution

    # load filter, get correct args
    filter_fn = images.get_filter(filter_name, implementation)
    filter_args = {
        "pixelator": {"blocksize": 80},  # higher -> more pixels
        "ascii": {"scale": 2},           # higher -> smaller resolution
        "blur": {"sigma": 5},            # higher -> more blur
    }

    # times and runs the function
    img, elapsed = utility.time_function(
        filter_fn,
        img,
        **filter_args.get(filter_name, {})
    )

    print(
        f'{"=" * 30}\n'
        f'{"File:":<20}{file}\n'
        f'{"Filter:":<20}{filter_name}\n'
        f'{"Implementation:":<20}{implementation}\n'
        f'{"Time:":<20}{elapsed:.4f}s\n'
        f'{"=" * 30}'
    )
    image_io.display(img)

    # print(img)


if __name__ == "__main__":
    main()
