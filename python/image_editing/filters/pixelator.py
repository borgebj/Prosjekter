from collections import Counter
import numpy as np


def python_pixelator(image: np.ndarray, blocksize=8, method="average") -> np.ndarray:
    height, width = image.shape[:2]

    # new output
    new_image = np.zeros_like(image)

    # go through all pixels
    # for each blocksize x blocksize:
    #   1. find most common color
    #   2. overwrite blocksize x blocksize with this
    for y in range(0, height - blocksize + 1, blocksize):
        for x in range(0, width - blocksize + 1, blocksize):

            colors = []

            # look through block for colors
            for dy in range(blocksize):
                for dx in range(blocksize):
                    pixel = image[y + dy, x + dx]
                    colors.append(tuple(pixel))

            if method == "average":
                r = sum(int(color[0]) for color in colors) // len(colors)
                g = sum(int(color[1]) for color in colors) // len(colors)
                b = sum(int(color[2]) for color in colors) // len(colors)

                new_pixel = (r, g, b)

            # most common
            else:
                new_pixel = Counter(colors).most_common(1)[0][0]

            # replace each pixel with its new pixel
            for dy in range(blocksize):
                for dx in range(blocksize):
                    new_image[y + dy, x + dx] = new_pixel

    return new_image


def numpy_pixelator(image: np.ndarray, blocksize=8) -> np.ndarray:
    # ready input, output
    array = np.array(image)

    height, width = array.shape[:2]

    # ensures image is compatible with blocksize
    height = height - height % blocksize
    width = width - width % blocksize
    array = array[:height, :width]

    # reshape image into blocks
    blocks = array.reshape(
        height // blocksize,
        blocksize,
        width // blocksize,
        blocksize,
        3
    )

    # average each block
    # e.g. turns 3x3 blocks into 1 pixel, avg color
    block_colors = blocks.mean(axis=(1, 3)).astype(np.uint8)

    # expand each block color back to blocksize x blocksize
    # e.g. takes the 1 pixel and 3x3's it
    result = np.repeat(
        np.repeat(block_colors, blocksize, axis=0),
        blocksize,
        axis=1
    )

    return result

