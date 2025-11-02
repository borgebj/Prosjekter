import numpy as np
from PIL import Image
import os

import images
from . import io
from . import utility


def main():
    filename = "test/rain.jpg"
    filter_name = "blur"

    img = io.read_image(filename)

    img = io.random_image(5, 5)

    # scaling
    img = utility.rescale(img, 3)

    # io.display(img)

    # load filter, run it
    filter_fn = images.get_filter(filter_name)
    img_filtered = filter_fn(img)

    io.display(img_filtered)

    # print(img)



