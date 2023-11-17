import numpy as np
import matplotlib.pyplot as plt


def implicit_heat(I, alpha, beta):
    """
    Applies the implicit heat equation to an image for image blurring/diffusion.

    Args:
        I (ndarray): The input image, either grayscale (2D array) or color (3D array).
        alpha (float): The diffusion coefficient in the x-direction.
        beta (float): The diffusion coefficient in the y-direction.

    Returns:
        ndarray: The image after applying implicit heat diffusion.
    """

    w, h = I.shape[:2]
    A = (1 + 2 * alpha) * np.eye(w) - alpha * (np.diag(np.ones(w - 1), k=1) + np.diag(np.ones(w - 1), k=-1))
    B = (1 + 2 * beta) * np.eye(h) - beta * (np.diag(np.ones(h - 1), k=1) + np.diag(np.ones(h - 1), k=-1))
    
    A_inv = np.linalg.inv(A)
    B_inv = np.linalg.inv(B)
    
    I_writable = I.copy() # Explicitly copy I to ensure it is writable

    if I_writable.ndim == 2:  
        for _ in range(10):
            I_writable = A_inv @ I_writable @ B_inv
    elif I_writable.ndim == 3:  
        for _ in range(10):
            for i in range(I_writable.shape[2]):
                I_writable[:, :, i] = A_inv @ I_writable[:, :, i] @ B_inv
    
    return I_writable


I = plt.imread("TP_scale_space/test_img.jpg")

processed_image = implicit_heat(I, 0.5, 0.5)

plt.imshow(processed_image, cmap="gray")
plt.show()