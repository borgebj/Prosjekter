import numpy as np
from PIL import Image


def read_image(filename: str) -> np.array:
    """Read an image file to a rgb array"""
    return np.asarray(Image.open(filename))


def random_image(width=320, height=180, dim=None) -> np.array:
    """Create a random image array of a given size"""
    if dim:
        return np.random.randint(0, 255, size=(height, width, dim), dtype=np.uint8)
    return np.random.randint(0, 255, size=(height, width), dtype=np.uint8)


def display(array: np.array):
    """Display image array on screen"""
    Image.fromarray(array).show()
