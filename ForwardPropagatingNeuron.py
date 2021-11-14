#
# Author: Lovsog
# Date: 2021.11.12 16:20
# Title: Forward propagating neuron (前向传播神经元)
#
#


import numpy as np
import math


class Neuron(object):
    def __init__(self):
        self.weights = np.array([1.0, 2.0, 3.0])  # 权重
        self.bias = 0.0  # 偏差

    def forward(self, inputs):
        """ Assuming that inputs and weights are 1-D numpy arrays and the bias is a number 假设输入和权重是一维 numpy 数组，偏差是一个数字 """
        a_cell_sum = np.sum(inputs * self.weights) + self.bias
        print(inputs)
        print(self.weights)
        print(a_cell_sum)
        result = 1.0 / (1.0 + math.exp(-a_cell_sum))  # This is the sigmoid activation function 这是 sigmoid 激活函数
        return result


neuron = Neuron()
output = neuron.forward(np.array([1, 1, 1]))
print(output)
