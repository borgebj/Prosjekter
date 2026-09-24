import images
import image_io
from python.image_editing import utility


def main():
    file = "cliff"
    filename = f"../images/{file}.jpg"
    filter_name = "pixelator"
    implementation = "numpy"

    img = image_io.read_image(filename)

    # mostly just noise, still interesting
    # img = image_io.random_image(1920, 1200)

    # scaling
    img = utility.rescale(img, scale=3)

    # load filter, run it
    filter_fn = images.get_filter(filter_name, implementation)
    img = filter_fn(img, blocksize=20)

    image_io.display(img)

    # print(img)


if __name__ == "__main__":
    main()
