from PIL import Image
import numpy as np


def rescale(img: np.array or Image.Image, scale: int = 10) -> Image.Image:
    """Rescale image using nearest neighbor"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    resized = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    return np.array(resized)


def printpic(image: np.ndarray):
    H, W = image.shape[:2]

    # padded image to handle borders (0 around edges)
    padded = np.pad(img, pad_width=1, mode="constant")
    H_p, W_p = padded.shape[:2]

    # copy of image with 0's
    blurred = np.zeros_like(img)

    print(kernel)
    print(W, H)
    print()

    for i in range(H_p):
        for j in range(W_p):
            pixel = padded[i, j]
            print(f"{pixel!s:5}", end=" ")
        print()

