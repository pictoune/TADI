import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage

def compute_grad(I):

    hy=np.array([[1/2,1,1/2]])
    hx=np.array([[-1/2,0,1/2]])
    Ix = signal.convolve2d(I, hy.T@hx)
    Iy = signal.convolve2d(I, hx.T@hy)

    mod = np.sqrt(np.square(Ix)+np.square(Iy))

    return mod,Ix,Iy

def enhance(I,t_step,k,sigma):
    w = I.shape[0]
    h = I.shape[1]
    im = I.copy()
    I1 = np.zeros((w+2,h+2))
    for _ in range(40):
        I1[1:w+1,1:h+1] = im
        I1[0,1:h+1] = im[0,:]
        I1[w+1,1:h+1] = im[w-1,:]
        I1[1:w+1,0] = im[:,0]
        I1[1:w+1,h+1] = im[:,h-1]
        Isig = ndimage.gaussian_filter(I1,sigma)
        mod, Ix, Iy = compute_grad(Isig)
        mod += 0.0000001
        lambda1 = np.exp(-mod**2/k**2)
        lambda2 = 0.2*lambda1
        a = (lambda1*Ix**2+lambda2*Iy**2)/mod**2
        b = (lambda1-lambda2)*Ix*Iy/mod**2
        c = (lambda2*(Ix)**2+lambda1*(Iy)**2)/mod**2
        for i in range(w):
            for j in range(h):
                im[i,j] = I1[i+1,j+1] + t_step*(-0.25*(b[i,j+1]+b[i+1,j+2])*I1[i,j+2]+0.5*(c[i+1,j+2]+c[i+1,j+1])*I1[i+1,j+2]+0.25*(b[i+2,j+1]+b[i+1,j+2])*I1[i+2,j+2]+0.5*(a[i,j+1]+a[i+1,j+1])*I1[i,j+1]-0.5*(a[i,j+1]+2*a[i+1,j+1]+a[i+2,j+1]+c[i+1,j]+2*c[i+1,j+1]+c[i+1,j+2])*I1[i+1,j+1]+0.5*(a[i+2,j+1]+a[i+1,j+1])*I1[i+2,j+1]+0.25*(b[i,j+1]+b[i+1,j])*I1[i,j+1]+0.5*(c[i+1,j]+c[i+1,j+1])*I1[i+1,j]-0.25*(b[i+2,j+1]+b[i+1,j])*I1[i+2,j])
    return im

fig = plt.figure()

I = plt.imread('img/synpic45657.jpg')

# fig.add_subplot(2,2,1)
# plt.imshow(enhance(I[:,:,0],0.25,100,0.1),cmap='gray')

# fig.add_subplot(2,2,2)
# plt.imshow(enhance(I[:,:,0],0.25,100,10),cmap='gray')

# fig.add_subplot(2,2,3)
# plt.imshow(enhance(I[:,:,0],0.25,100,100),cmap='gray')

# fig.add_subplot(2,2,4)
# plt.imshow(enhance(I[:,:,0],0.25,100,1000),cmap='gray')

plt.imshow(enhance(I[:,:,0],0.25,100,100),cmap='gray')
plt.show()