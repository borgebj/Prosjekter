from pathlib import Path
from PIL import Image
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_editing.filters import ascii, greyscale, pixelator, sepia

SOURCE = ROOT / 'image_editing' / 'images' / 'mexico.jpg'
OUTPUT_DIR = Path(__file__).resolve().parent / 'generated'
OUTPUT_DIR.mkdir(exist_ok=True)

source = np.asarray(Image.open(SOURCE))
Image.fromarray(source).save(OUTPUT_DIR / '01_original.png')
Image.fromarray(greyscale.numpy_greyscale(source)).save(OUTPUT_DIR / '02_greyscale.png')
Image.fromarray(sepia.numpy_sepia(source)).save(OUTPUT_DIR / '03_sepia.png')
Image.fromarray(pixelator.numpy_pixelator(source, blocksize=20)).save(OUTPUT_DIR / '04_pixelated.png')
Image.fromarray(ascii.numpy_ascii(source, scale=7)).save(OUTPUT_DIR / '05_ascii.png')

placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
placeholder[:] = (70, 90, 110)
Image.fromarray(placeholder).save(OUTPUT_DIR / '06_blur_placeholder.png')

print(f'Generated showcase images from {SOURCE}')
