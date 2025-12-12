import numpy as np
from typing import Optional


def python_sepia(image: np.ndarray) -> np.ndarray:
    sepia_image = np.empty_like(image)
    height = image.shape[0]
    width = image.shape[1]
    depth = image.shape[2]

    sepia_matrix = [
        [0.393, 0.769, 0.189],  # 0
        [0.349, 0.686, 0.168],  # 1
        [0.272, 0.534, 0.131],  # 2
    ]     # 0    # 1    # 2

    for i in range(height):
        for j in range(width):
            red, green, blue = image[i, j]

            # k = current color channel
            for k in range(depth):
                pixel = red * sepia_matrix[k][0] + \
                        green * sepia_matrix[k][1] + \
                        blue * sepia_matrix[k][2]

                sepia_image[i ,j ,k] = min(pixel, 255)

    return sepia_image.astype("uint8")


def numpy_sepia(image: np.ndarray, k: Optional[float] = 1) -> np.ndarray:
    sepia_image = np.empty_like(image)

    if not 0.0 <= k <= 1.0:
        raise ValueError(f"K msut be between [0-1], got {k=}")

    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],  # 0
        [0.349, 0.686, 0.168],  # 1
        [0.272, 0.534, 0.131],  # 2
    ])    # 0    # 1    # 2

    # multiplies image with sepia-matrix transposed
    sepia_image = image @ sepia_matrix.T

    # adds altered sepia and altered original with K together to create a mix
    # image only tunes when specified - 1 = default
    if 0.0 <= k < 1.0:
        sepia_image = (sepia_image * k) + image * (1 - k)

    # ensures no value exceeds 255
    sepia_image = sepia_image.clip(0, 255)

    # Returns image as right type
    return sepia_image.astype("uint8")

