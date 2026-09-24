from PIL import Image
import numpy as np
import time
import random
import os


def rescale(img: np.array or Image.Image, scale: int = 10) -> Image.Image:
    """Rescale image using nearest neighbor"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    resized = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    return np.array(resized)


def time_function(function, *args, **kwargs):
    """Runs and times a given function"""
    start = time.perf_counter()

    result = function(*args, **kwargs)

    elapsed = time.perf_counter() - start

    return result, elapsed


def random_image(directory="../images"):
    """Returns a random image filename from image directory"""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files = [
        file for file in os.listdir(directory)
        if file.lower().endswith(extensions)
    ]

    if not files:
        raise FileNotFoundError(f'No images found in {directory}')

    return random.choice(files)
