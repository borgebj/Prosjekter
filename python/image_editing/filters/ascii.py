from PIL import Image, ImageDraw, ImageFont
import numpy as np


def python_ascii(image: np.ndarray, scale=2) -> np.ndarray:
    characters = "██▓▓▒▒░░  "

    height, width = image.shape[:2]

    # character dimensions
    char_width = 8
    char_height = 8

    # ascii resolution
    new_width = width // scale
    new_height = height // scale

    # resize image to ascii resolution
    image = Image.fromarray(image)
    image = image.resize((new_width, new_height))

    # output image
    ascii_image = Image.new(
        "L",
        (new_width * char_width, new_height * char_height),
        "black"
    )

    draw = ImageDraw.Draw(ascii_image)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", char_height)

    # goes through each (new) pixel / ascii position
    for y in range(new_height):
        for x in range(new_width):

            # get pixel and greyscale it
            r, g, b = image.getpixel((x, y))
            brightness = (
                r * 0.21 +
                g * 0.72 +
                b * 0.07
            )

            # get specific ascii for that brightness
            index = int(brightness / 256 * len(characters))
            character = characters[index]

            # adds this to the PIL image
            draw.text(
                (x * char_width, y * char_height),
                character,
                fill="white",
                font=font
            )

    return np.array(ascii_image)


def numpy_ascii(image: np.ndarray, scale=2) -> np.ndarray:
    characters = "██▓▓▒▒░░  "

    height, width = image.shape[:2]

    # character dimensions
    char_width = 8
    char_height = 8

    # ascii resolution
    new_width = width // scale
    new_height = height // scale

    image = Image.fromarray(image)
    image = image.resize((new_width, new_height))
    image = np.array(image)

    # greyscale the image
    greyscale = (
            image[..., 0] * 0.21 +
            image[..., 1] * 0.72 +
            image[..., 2] * 0.07
    ).astype(np.uint8)

    # turns pixel-brightnesses into character-indices
    indices = (greyscale / 256 * len(characters)).astype(int)

    # render each character once
    font = ImageFont.truetype(
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        char_height
    )

    glyphs = []
    for character in characters:
        glyph = Image.new(
            "L",
            (char_width, char_height),
            0
        )

        draw = ImageDraw.Draw(glyph)
        draw.text(
            (0, 0),
            character,
            fill=255,
            font=font
        )

        glyphs.append(np.array(glyph))

    glyphs = np.array(glyphs)

    # select correct glyph for each ascii position
    output = glyphs[indices]

    # rearrange into one large image
    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(
        new_height * char_height,
        new_width * char_width
    )

    return output