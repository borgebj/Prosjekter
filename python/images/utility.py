from PIL import Image
import numpy as np


def rescale(img: np.array or Image.Image, scale: int = 10) -> Image.Image:
    """Rescale image using nearest neighbor"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    resized = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    return np.array(resized)