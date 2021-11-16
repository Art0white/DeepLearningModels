#
# Author: Lovsog
# Date: 2021.11.16 21:24
# Title: InterpretTheImage (计算机如何解释图像)
#

import os
os.environ['TF_CPP_MIN_LOG_LEVLE'] = '2'
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# 图像可视化
image = imread('img/testimg.png', as_gray=True)
image = resize(image, (28, 28), mode='reflect')
print('This image is: ', type(image),
      'with dimensions:', image.shape)
plt.imshow(image, cmap='gray')

# 图像像素值可视化(黑 0.00, 白 1.00)
def visualize_input(img, ax):
      ax.imshow(img, cmap='gray')
      width, height = img.shape
      thresh = img.max()/2.5
      for x in range(width):
            for y in range(height):
                  ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                  horizontalalignment='center',
                  verticalalignment='center',
                  color='white' if img[x][y] < thresh else 'black')

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
visualize_input(image, ax)


plt.show()
# 不必 import matplotlib.pyplot as plt 后面加 %matplotlib inline, 在最后加 plt.show 即可
