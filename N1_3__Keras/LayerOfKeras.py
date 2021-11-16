#
# Author: Lovsog
# Date: 2021.11.16 18:04
# Title: LayerOfKeras (Keras模型中的层)
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 序列
from tensorflow.python.keras.models import Sequential
# 全连接层 激活层 过滤层
from tensorflow.python.keras.layers.core import Dense, Activation, Flatten


# Creating the Sequential model
model = Sequential()

# Layer 1 - Adding a flatten layer
# 第一层: 输入维度为(32, 32, 3), 输出维度为(3072 = 32 * 32 * 3)
# 32 * 32 的二维图片, 3 代表 RGB 对应三原色, 三色图叠加产生彩色图
model.add(Flatten(input_shape=(32, 32, 3)))

# Layer 2 - Adding a fully connected layer
# 第二层: 输入为第一层输出, 输出维度为 100
model.add(Dense(100))

# Layer 3 - Adding a ReLU activation layer
model.add(Activation('relu'))

# Layer 4 - Adding a fully connected layer
model.add(Dense(60))

# Layer 5 - Adding an ReLU activation layer
model.add(Activation('relu'))
