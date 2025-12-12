import numpy as np
from PIL import Image
import os

import images
from . import io
from . import utility


def main():
    file = "mexico"
    filename = f"test/{file}.jpg"
    filter_name = "greyscale"
    implementation = "numpy"

    img = io.read_image(filename)

    # img = io.random_image(1920, 1200)

    # scaling
    img = utility.rescale(img, scale=3)

    # load filter, run it
    filter_fn = images.get_filter(filter_name, implementation)
    img = filter_fn(img)

    io.display(img)

    print(img)



