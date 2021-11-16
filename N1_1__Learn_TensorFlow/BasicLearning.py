#
# Author: Lovsog
# Date: 2021.11.13 10:14
# Title: BasicLearing (TensorFlow基础前半部分, 后半部分回到Install.py)
#

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# A is an int32 tensor with rank = 0
A = tf.constant(123)
print(A)

# B is an int32 tensor with dimension of 1 ( rank = 1 )
B = tf.constant([123, 456, 789])
print(B)

# C is an int32 2- dimensional tensor
C = tf.constant([[123, 456, 789], [222, 333, 444]])
print(C)

