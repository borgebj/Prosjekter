import numpy as np


def python_greyscale(image: np.ndarray) -> np.ndarray:
    gray_image = np.empty_like(image)
    height = image.shape[0]
    width = image.shape[1]
    depth = image.shape[2]

    # apply greyscale to each color pixel
    for i in range(height):
        for j in range(width):
            red = image[i, j, 0] * 0.21
            green = image[i, j, 1] * 0.72
            blue = image[i, j, 2] * 0.07
            pixel = (red + green + blue)

            for k in range(depth):
                gray_image[i, j, k] = pixel

    return gray_image.astype("uint8")


def numpy_greyscale(image: np.ndarray) -> np.ndarray:
    gray_image = np.empty_like(image)

    # gets every pixel from each pixel-channl and applies weights
    red = image[..., 0] * 0.21      # new red value
    green = image[..., 1] * 0.72    # new green value
    blue = image[..., 2] * 0.07     # new blue value

    # assigns the new values tp each pixel with weighted sum of each color
    gray_image[..., 0] = red + green + blue
    gray_image[..., 1] = red + green + blue
    gray_image[..., 2] = red + green + blue

    # returns and ensures correct datatype
    return gray_image.astype("uint8")