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

# Load a color image in grayscale
image = imread('img/testimg.jpg', as_gray=True)
image = resize(image, (80, 50), mode='reflect')
print('This image is: ', type(image),
      'with dimensions:', image.shape)
plt.imshow(image, cmap='gray')