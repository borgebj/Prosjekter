from PIL import Image, ImageDraw, ImageFont

image = Image.open("images/hackerman.jpg").convert("L")

# ASCII characters, dark -> light
characters = "█▓▒░ "
characters = "MMMMmmmmm....... "

# How much detail
scale = 2

# Character dimensions
char_width = 8
char_height = 16

# Character aspect ratio correction
# Characters are roughly twice as tall as they are wide.
correction = char_width / char_height

# Resize image to ASCII resolution
new_width = image.width // scale
new_height = int(image.height // scale * correction)

image = image.resize((new_width, new_height))

pixels = image.load()

# Create output image
ascii_image = Image.new(
    "RGB",
    (new_width * char_width, new_height * char_height),
    "black"
)

draw = ImageDraw.Draw(ascii_image)

font = ImageFont.truetype(
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    char_height
)

for y in range(new_height):
    for x in range(new_width):

        brightness = pixels[x, y]

        index = int(
            brightness / 256 * len(characters)
        )
        character = characters[index]

        draw.text(
            (x * char_width, y * char_height),
            character,
            fill="white",
            font=font
        )

ascii_image.show()