import numpy as np
import matplotlib.pyplot as plt

def heat(I,t_step):
    w = I.shape[0]
    h = I.shape[1]
    im = I.copy()
    I1 = np.zeros((w+2,h+2))
    for _ in range(20):
        I1[1:w+1,1:h+1] = im
        I1[0,1:h+1] = im[0,:]
        I1[w+1,1:h+1] = im[w-1,:]
        I1[1:w+1,0] = im[:,0]
        I1[1:w+1,h+1] = im[:,h-1]
        for i in range(w):
            for j in range(h):
                im[i,j] = I1[i+1,j+1] + t_step*(I1[i+2,j+1]-2*I1[i+1,j+1]+I1[i,j+1]+I1[i+1,j+2]-2*I1[i+1,j+1]+I1[i+1,j])
    return im

fig = plt.figure()

I = plt.imread('img/synpic45657.jpg')

# fig.add_subplot(2,2,1)
# plt.title(r'$\Delta t = 0.1$')
# plt.imshow(heat(I[:,:,0],0.1),cmap='gray')

# fig.add_subplot(2,2,2)
# plt.title(r'$\Delta t = 0.22$')
# plt.imshow(heat(I[:,:,0],0.22),cmap='gray')

# fig.add_subplot(2,2,3)
# plt.title(r'$\Delta t = 0.27$')
# plt.imshow(heat(I[:,:,0],0.27),cmap='gray')

# fig.add_subplot(2,2,4)
# plt.title(r'$\Delta t = 0.4$')
# plt.imshow(heat(I[:,:,0],0.4),cmap='gray')

plt.imshow(heat(I[:,:,0],0.22),cmap='gray')
plt.show()
# plt.imshow(I[:,:,0],cmap='gray')
# plt.show()