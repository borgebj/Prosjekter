import numpy as np
from utility import rescale


# THIS IS W.I.P

def gaussian_kernel_1d(kernel_size, sigma=1.0) -> np.ndarray:
    """Generates a Gaussian kernel"""
    radius = kernel_size // 2

    kernel = np.zeros(kernel_size)

    # 1D gaussian function for normalization
    for x in range(-radius, radius + 1):
        exponent = -(x ** 2) / (2 * sigma ** 2)
        kernel[x + radius] = np.exp(exponent)

    # normalize kernel to ensure sum is 1
    kernel /= kernel.sum()

    return kernel

def gaussian_kernel(kernel_size, sigma=1.0) -> np.ndarray:
    """Generates a Gaussian kernel"""
    radius = kernel_size // 2

    kernel = np.zeros((kernel_size, kernel_size))

    # calculate Gaussian values for each position in the kernel
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):

            # gaussian function:
            #   G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            kernel[y + radius, x + radius] = np.exp(exponent)

    # normalize kernel to ensure sum is 1
    kernel /= kernel.sum()

    return kernel


def python_blur(image: np.ndarray, sigma=1.0) -> np.ndarray:
    """Gaussian blur using pure Python"""
    height, width = image.shape[:2]

    # calculate kernel size based on sigma
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # create kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    radius = kernel_size // 2

    # output image
    new_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):

            r = g = b = 0.0

            # get surrounding pixels
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):

                    neighbor_y = y + dy
                    neighbor_x = x + dx

                    # clamp neighbor coordinates to image boundaries - essentially replicating edge pixels
                    neighbor_y = max(0, min(neighbor_y, height - 1))
                    neighbor_x = max(0, min(neighbor_x, width - 1))

                    # get pixel and weight
                    pixel = image[neighbor_y, neighbor_x]
                    weight = kernel[dy + radius, dx + radius]

                    # add weighted pixel to total
                    r += pixel[0] * weight
                    g += pixel[1] * weight
                    b += pixel[2] * weight

            new_image[y, x] = (r, g, b)

    return new_image


def numpy_blur(image: np.ndarray, sigma=1.0) -> np.ndarray:
    """Gaussian blur using numpy operations on inner part of the image"""
    height, width = image.shape[:2]

    # calculate kernel size based on sigma
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # create kernel (average)
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    radius = kernel_size // 2

    # convert image to float for precision during convolution
    img_float = image.astype(np.float32)

    # --------------------
    # horizontal conv
    # --------------------

    # pad left/right
    padded = np.pad(
        img_float,
        ((0, 0), (radius, radius), (0, 0)),
        mode='edge'
    )

    horizontal = np.zeros_like(img_float)

    # convolution using numpy operations
    for x in range(width):
        region = padded[:, x:x + kernel_size, :]  # neighboring columns for each pixel

        # applies 1d kernel horizontally
        horizontal[:, x, :] = np.sum(
            region * kernel[None, :, None],
            axis=1
        )

    # --------------------
    # vertical conv
    # --------------------

    # pad top/bottom
    padded = np.pad(
        horizontal,
        ((radius, radius), (0, 0), (0, 0)),
        mode='edge'
    )

    output = np.zeros_like(img_float)

    for y in range(height):

        region = padded[y:y + kernel_size, :, :] # neighboring rows for each pixel

        # applies 1d kernel vertically
        output[y, :, :] = np.sum(
            region * kernel[:, None, None],
            axis=0
        )

    return np.clip(output, 0, 255).astype(np.uint8)