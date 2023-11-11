import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage


def compute_grad(I):
    """
    Compute the gradient of an image.

    I: The input image.

    Returns:
        mod: The magnitude of the gradient.
        Ix: The gradient along the x-axis.
        Iy: The gradient along the y-axis.
    """
    hy = np.array([[1 / 2, 1, 1 / 2]])
    hx = np.array([[-1 / 2, 0, 1 / 2]])
    Ix = signal.convolve2d(I, hy.T @ hx)
    Iy = signal.convolve2d(I, hx.T @ hy)
    mod = np.sqrt(np.square(Ix) + np.square(Iy))

    return mod, Ix, Iy


def enhance(I, t_step, k, sigma):
    """
    Enhance an image using the Perona-Malik diffusion equation.

    Args:
        I: The input image.
        t_step: The time step for the diffusion process.
        k: The diffusion coefficient.
        sigma: The standard deviation for Gaussian filtering.

    Returns:
        im: The enhanced image.
    """
    im = I.copy()
    W, H = I.shape
    I1 = np.zeros((W + 2, H + 2))

    for _ in range(40):
        I1 = np.pad(im, pad_width=1, mode="edge")
        Isig = ndimage.gaussian_filter(I1, sigma)
        mod, Ix, Iy = compute_grad(Isig)
        mod += 0.0000001
        lambda1 = np.exp(-(mod**2) / k**2)
        lambda2 = 0.2 * lambda1
        a = (lambda1 * Ix**2 + lambda2 * Iy**2) / mod**2
        b = (lambda1 - lambda2) * Ix * Iy / mod**2
        c = (lambda2 * (Ix) ** 2 + lambda1 * (Iy) ** 2) / mod**2

        for i in range(W):
            for j in range(H):
                im[i, j] = I1[i + 1, j + 1] + t_step * (
                    -0.25 * (b[i, j + 1] + b[i + 1, j + 2]) * I1[i, j + 2]
                    + 0.5 * (c[i + 1, j + 2] + c[i + 1, j + 1]) * I1[i + 1, j + 2]
                    + 0.25 * (b[i + 2, j + 1] + b[i + 1, j + 2]) * I1[i + 2, j + 2]
                    + 0.5 * (a[i, j + 1] + a[i + 1, j + 1]) * I1[i, j + 1]
                    - 0.5
                    * (
                        a[i, j + 1]
                        + 2 * a[i + 1, j + 1]
                        + a[i + 2, j + 1]
                        + c[i + 1, j]
                        + 2 * c[i + 1, j + 1]
                        + c[i + 1, j + 2]
                    )
                    * I1[i + 1, j + 1]
                    + 0.5 * (a[i + 2, j + 1] + a[i + 1, j + 1]) * I1[i + 2, j + 1]
                    + 0.25 * (b[i, j + 1] + b[i + 1, j]) * I1[i, j + 1]
                    + 0.5 * (c[i + 1, j] + c[i + 1, j + 1]) * I1[i + 1, j]
                    - 0.25 * (b[i + 2, j + 1] + b[i + 1, j]) * I1[i + 2, j]
                )

    return im


fig = plt.figure()

I = plt.imread("TP_scale_space/test_img.jpg")

plt.imshow(enhance(I[:, :, 0], 0.25, 100, 100), cmap="gray")
plt.show()
