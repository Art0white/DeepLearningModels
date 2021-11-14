#
# Author: Lovsog
# Date: 2021.11.12 16:54
# Title: Install (安装TensorFlow)
#

import tensorflow as tf
import tensorboard
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.disable_eager_execution()

hello_constant = tf.constant('Hello World!', name='hello_constant')

# tensorboard --logdir = path/to/log-directory 使用TensorFlow可视化图像运行指令


#
# Author: Lovsog
# Date: 2021.11.13 18:27
# Title: 基础后半部分
#

# TensorFlow 2.0以上, sess = tf.Session() 不可用，改为 sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    output = sess.run(hello_constant)
    print(output)

constant_x = tf.constant(5, name='constant_x')
variable_y = tf.Variable(constant_x + 5, name='variable_y')
print(variable_y)
# 输出<tf.Variable 'variable_y:0' shape=() dtype=int32>, 而不是我们想要的结果10
# 更改如下

# init = tf.global_varialbes_initializer() 新版本不可以该方法
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(sess.run(variable_y))
# 成功输入10
