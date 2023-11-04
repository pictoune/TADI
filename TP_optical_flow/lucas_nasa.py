from matplotlib import pyplot as plt
import numpy as np
from middlebury import computeColor
from lucas import lucas,lucas_gaussian,quiver
from middlebury import computeColor

I1 = plt.imread("../data/nasa/nasa9.png")
I2 = plt.imread("../data/nasa/nasa10.png")

fig = plt.figure(figsize=(15,4))

for i,n in enumerate([2,10,24,50]):
    fig.add_subplot(2, 4, i+1)
    fig.tight_layout(pad=0.2)
    plt.imshow(computeColor(lucas(I1,I2,n)))
    plt.title("n = %i" % n, fontsize=16)
    fig.add_subplot(2, 4, i+5)
    fig.tight_layout(pad=0.2)
    plt.imshow(computeColor(lucas_gaussian(I1,I2,n,5)))
    plt.title("n = %i" % n, fontsize=16)
plt.show()

quiver(lucas_gaussian(I1,I2,24,5),'Best flow as a vector field obtained with the Lucas-Kanade method',15)
quiver(lucas_gaussian(I1,I2,24,5),'Best flow as a normalized vector field obtained with the Lucas-Kanade method',40,norm=True)