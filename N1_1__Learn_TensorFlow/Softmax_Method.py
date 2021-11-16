#
# Author: Lovsog
# Date: 2021.11.14 18:38
# Title: Softmax_Method (TensorFlow中的softmax函数)
#

# softmax 函数将其输入(被称为 logit 或者 logitechscore )转换为 0 到 1 之间的值, 并对其输出进行归一化，使其总和为 1
# logit 是指一个事件发生与不发生的概率比值的对数。

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 用占位符(placeholder)必须加这句
tf.compat.v1.disable_eager_execution()

logit_data = [2.0, 1.0, 0.1]
logits = tf.compat.v1.placeholder(tf.float32)
softmax = tf.nn.softmax(logits)

with tf.compat.v1.Session() as sess:
    output = sess.run(softmax, feed_dict={logits: logit_data})
    print(output)


# 在机器学习中, 我们通常用数学函数来定义一个模型的好坏, 这个函数叫作 损失函数、成本函数或者目标函数。
# 用于确定模型损失的一个很常见的函数叫 交叉熵 损失。

# test1
test1_x = tf.compat.v1.constant([[1, 1, 1], [1, 1, 1]])

with tf.compat.v1.Session() as sess:
    print(sess.run(tf.reduce_sum([1, 2, 3])))      # returns 6
    print(sess.run(tf.reduce_sum(test1_x, 0)))     # sum along x axis, prints [2, 2, 2]


# test2
softmax_data = [0.1, 0.5, 0.4]
onehot_data = [0.0, 1.0, 0.0]

softmax = tf.compat.v1.placeholder(tf.float32)
onehot_encoding = tf.compat.v1.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(tf.multiply(onehot_encoding, tf.compat.v1.log(softmax)))

cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.compat.v1.log(softmax), labels=onehot_encoding)

# 手动计算交叉熵损失
with tf.compat.v1.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={softmax:softmax_data, onehot_encoding:onehot_data}))
    # returns 0.6931472
    print(sess.run(cross_entropy_loss, feed_dict={softmax: softmax_data, onehot_encoding: onehot_data}))
    # returns 0.6931472
