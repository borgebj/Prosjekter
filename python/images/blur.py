import numpy as np


def python_blur(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3)) / 9
    H, W = image.shape[:2]

    # padded image to handle borders (0 around edges)
    padded = np.pad(img, pad_width=1, mode="constant")
    H_p, W_p = padded.shape[:2]

    # copy of image with 0's
    blurred = np.zeros_like(img)

    print(kernel)
    print(W, H)
    print()
    # print(padded)

    for i in range(H_p):
        for j in range(W_p):
            pixel = padded[i, j]
            # print(f"{pixel!s:5}", end=" ")
        # print()

    return img


def numpy_blur(image: np.ndarray) -> np.ndarray:
    return NotImplementedError