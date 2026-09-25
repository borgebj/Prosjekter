# Image Editing Experiments

This project is a small collection of image-processing experiments built in Python. The focus is on understanding how image filters work, how pixel data is transformed, and how simple visual effects can be created using NumPy and Pillow.

## Included concepts

- Grayscale conversion
- Sepia tone effects
- Pixelation
- Blur
- ASCII-style image rendering
- Basic image IO and display utilities

## Tech stack

- Python
- NumPy
- Pillow

## Project structure

```text
image_editing/
├── blur.py
├── game.py
├── README.md
├── filters/
│   ├── ascii.py
│   ├── greyscale.py
│   ├── image_io.py
│   ├── images.py
│   ├── main.py
│   ├── pixelator.py
│   ├── sepia.py
│   └── utility.py
└── ...
```

## Example usage

From the `python` directory, install dependencies and run the demo:

```bash
pip install -r requirements.txt
cd image_editing/filters
python main.py
```

The project is structured around a reusable image filter pipeline. Each filter can be selected and applied to an image, and the code is designed to make experimentation easy.

## What this project demonstrates

This is a good portfolio project because it shows that I can work with:

- Image arrays and pixel manipulation
- Computer vision concepts
- Python libraries for practical visual processing
- Small experimental project design that is easy to extend

## Possible next steps

- Add more filters such as sharpen, vignette, contrast adjustments, and edge detection
- Add a CLI with selectable filter options
- Add example output images for each effect
- Improve the project writing so it is easier to run and understand for a recruiter or collaborator