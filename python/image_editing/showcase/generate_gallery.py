from pathlib import Path
from PIL import Image
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
FILTERS_DIR = ROOT / "filters"

if str(FILTERS_DIR) not in sys.path:
    sys.path.insert(0, str(FILTERS_DIR))

import ascii, blur, greyscale, pixelator, sepia

SOURCE = ROOT / "images" / "mexico.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / 'generated'
OUTPUT_DIR.mkdir(exist_ok=True)

source = np.asarray(Image.open(SOURCE))
Image.fromarray(source).save(OUTPUT_DIR / '01_original.png')
Image.fromarray(greyscale.numpy_greyscale(source)).save(OUTPUT_DIR / '02_greyscale.png')
Image.fromarray(sepia.numpy_sepia(source)).save(OUTPUT_DIR / '03_sepia.png')
Image.fromarray(pixelator.numpy_pixelator(source, blocksize=20)).save(OUTPUT_DIR / '04_pixelated.png')
Image.fromarray(ascii.numpy_ascii(source, scale=7)).save(OUTPUT_DIR / '05_ascii.png')
Image.fromarray(blur.numpy_blur(source, sigma=5)).save(OUTPUT_DIR / '06_blur.png')

print(f'Generated showcase images from {SOURCE}')
