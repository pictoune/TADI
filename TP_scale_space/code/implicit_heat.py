import numpy as np
import matplotlib.pyplot as plt

def implicit_heat(I,alpha,beta):
    w = I.shape[0]
    h = I.shape[1]
    im = I.copy()
    A = (1+2*alpha)*np.eye(w)-alpha*(np.diag(np.ones(w-1),k=1)+np.diag(np.ones(w-1),k=-1))
    B = (1+2*beta)*np.eye(h)-beta*(np.diag(np.ones(h-1),k=1)+np.diag(np.ones(h-1),k=-1))
    for _ in range(10):
        im = np.linalg.inv(A)@im@np.linalg.inv(B)
    return im

I = plt.imread('img/fingerprint-small.jpg')
plt.imshow(implicit_heat(I, 0.5,0.5),cmap='gray')
plt.show()
# plt.imshow(I,cmap='gray')
# plt.show()