#
# Author: Lovsog
# Date: 2021.11.16 19:59
# Title: HandwrittenDigitRecognition (基于Keras和MNIST的手写数字识别)
#

# Import Numpy, keras and MNIST data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.utils import np_utils

# Retrieving the training and test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# test 1: 训练和测试数据的检索
# 数据降维
# 训练数据集
print('X_train shape:', X_train.shape)  # X_train shape: (60000, 28, 28)
# 测试数据集
print('X_test shape:', X_test.shape)  # X_test shape: (10000, 28, 28)

print('y_train shape:', y_train.shape)  # y_train shape: (60000,)

print('y_test shape:', y_test.shape)  # y_test shape: (10000,)


# test 2: 训练数据的可视化
# import matplotlib.pyplot as plt
# %matplotlib imline    这里用这句话报错, 用plt.show()来展示图表

# Displaying a training image by its index in the MNIST set
def display_digit(index):
    label = y_train[index].argmax(axis=0)
    image = X_train[index]
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()


# Displaying the first (index 0) training image
display_digit(0)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Train the matrix shape", X_train.shape)
print("Test the matrix shape", X_test.shape)

# One Hot encoding of labels.
from tensorflow.python.keras.utils.np_utils import to_categorical
print(y_train.shape)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(y_train.shape)


# test 3: 创建神经网络
def build_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))  # An "activation" is just a non-linear function that is applied to the output of the above layer. In this case, with a "rectified liner unit", we perform clamping on all values below 0 to 0.
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

# Building the model
model = build_model()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# test 4: 训练神经网络
# 新版本把 nb_epoch=4 改为 epochs=4
model.fit(X_train, y_train, batch_size=128, epochs=4, verbose=1, validation_data=(X_test, y_test))


# test 5: 测试
# 好的结果是将获得高于 95% 的准确率
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1, sample_weight=None)
# Printing the result
print('Test score:', score[0])
print('Test accuracy:', score[1])
