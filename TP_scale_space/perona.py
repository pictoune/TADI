import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compute_grad(I):

    hy=np.array([[1/2,1,1/2]])
    hx=np.array([[-1/2,0,1/2]])
    Ix = signal.convolve2d(I, hy.T@hx)
    Iy = signal.convolve2d(I, hx.T@hy)

    mod = np.sqrt(np.square(Ix)+np.square(Iy))

    return mod

def perona(I,t_step,K,alpha):
    def g(x):
        return 1/(1+(x/K)**(1+alpha))
    w = I.shape[0]
    h = I.shape[1]
    im = I.copy()
    I1 = np.zeros((w+2,h+2))+200
    for _ in range(30):
        I1[1:w+1,1:h+1] = im
        I1[0,1:h+1] = im[0,:]
        I1[w+1,1:h+1] = im[w-1,:]
        I1[1:w+1,0] = im[:,0]
        I1[1:w+1,h+1] = im[:,h-1]
        grad = compute_grad(I1)
        for i in range(w):
            for j in range(h):
                # im[i,j] = I1[i+1,j+1] + t_step*(g(np.abs(I1[i,j+1]-I1[i+1,j+1])) * (I1[i,j+1]-I1[i+1,j+1]) + g(np.abs(I1[i+2,j+1]-I1[i+1,j+1])) * (I1[i+2,j+1]-I1[i+1,j+1]) + g(np.abs(I1[i+1,j+2]-I1[i+1,j+1])) * (I1[i+1,j+2]-I1[i+1,j+1]) + g(np.abs(I1[i+1,j]-I1[i+1,j+1])) * (I1[i+1,j]-I1[i+1,j+1]))
                im[i,j] = I1[i+1,j+1] + t_step*(g(1/2*(grad[i+1,j+1]+grad[i+2,j+1])) * (I1[i,j+1]-I1[i+1,j+1]) + g(1/2*(grad[i+1,j+1]+grad[i,j+1])) * (I1[i+2,j+1]-I1[i+1,j+1]) + g(1/2*(grad[i+1,j+1]+grad[i+1,j+2])) * (I1[i+1,j+2]-I1[i+1,j+1]) + g(1/2*(grad[i+1,j+1]+grad[i+1,j])) * (I1[i+1,j]-I1[i+1,j+1]))
    return im

# fig = plt.figure()
I = plt.imread('img/synpic45657.jpg')

# fig.add_subplot(2,2,1)
# plt.imshow(perona(I[:,:,0],0.22,15,0.1),cmap='gray')

# fig.add_subplot(2,2,2)
# plt.imshow(perona(I[:,:,0],0.22,15,5),cmap='gray')

# fig.add_subplot(2,2,3)
# plt.imshow(perona(I[:,:,0],0.22,15,20),cmap='gray')

# fig.add_subplot(2,2,4)
# plt.imshow(perona(I[:,:,0],0.22,15,70),cmap='gray')

plt.imshow(perona(I[:,:,0],0.22,15,20),cmap='gray')
plt.show()
# plt.imshow(I,cmap='gray')
# plt.show()