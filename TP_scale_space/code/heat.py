import numpy as np
import matplotlib.pyplot as plt


def heat(I, t_step):
    """
    Applies a heat diffusion process to an image `I` for a fixed number of iterations.
    This process is akin to applying a discrete version of the heat equation to the image.
    The result is a smoothed version of the original image.

    Args:
        I: The input image.
        t_step: The time step for the heat diffusion.

    Returns:
        The image after applying heat diffusion.
    """
    im = I.copy()

    for _ in range(20):
        I1 = np.pad(im, 1, mode="edge")
        im = I1[1:-1, 1:-1] + t_step * (
            I1[2:, 1:-1]
            - 2 * I1[1:-1, 1:-1]
            + I1[:-2, 1:-1]
            + I1[1:-1, 2:]
            - 2 * I1[1:-1, 1:-1]
            + I1[1:-1, :-2]
        )

    return im


fig = plt.figure()

I = plt.imread("TP_scale_space/test_img.jpg")

plt.imshow(heat(I[:, :, 0], 0.22), cmap="gray")
plt.show()