import numpy as np

def blur(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3)) / 9
    H, W = img.shape[:2]

    padded = np.pad(img, pad_width=1, mode="constant")
    H_p, W_p = padded.shape[:2]


    print(kernel)
    print(H, W)
    print()
    # print(padded)

    for i in range(H_p):
        for j in range(W_p):
            pixel = padded[i, j]
            print(f"{pixel!s:5}", end=" ")
        print()

    return padded