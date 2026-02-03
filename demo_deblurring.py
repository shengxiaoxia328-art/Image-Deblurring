import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot as plt


im = Image.open('test_images/1024_books_original.png') 
img = np.array(im)
img = img.astype(np.float64) / 255

m, n = img.shape
img[42,0:n-1] = 0

plt.figure(1)
plt.axis('off')
plt.gray()
plt.imshow(img)
plt.show()
plt.imsave('meaning.png', img)