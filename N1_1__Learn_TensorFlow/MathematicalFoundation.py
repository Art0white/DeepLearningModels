#
# Author: Lovsog
# Date: 2021.11.13 18:32
# Title: Mathematical foundation (TensorFlow中的数学基础)
#

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

init = tf.compat.v1.global_variables_initializer()

# 加法
with tf.compat.v1.Session() as sess:
    add_x = tf.add(1, 2, name=None)
    print(sess.run(add_x))
# 返回 3

# 减法
with tf.compat.v1.Session() as sess:
    sub_x = tf.subtract(1, 2, name=None)
    print(sess.run(sub_x))
# 返回 -1

# 乘法
with tf.compat.v1.Session() as sess:
    mult_x = tf.multiply(2, 5, name=None)
    print(sess.run(mult_x))
# 返回 10


# 使用API————tf.placeholder(), 也就是占位符, 来处理非常量值. 还有feed_dict类型
# 哦对了! 新版本应该用tf.compat.v1.placeholder()

# test1 : 在会话运行前, 将字符串设置给x
# 这里会有一个 “RuntimeError: tf.placeholder() is not compatible with eager execution." 的报错
# 是因为TensorFlow 2.0 及以上版本, 默认情况下开启了紧急执行模式，即定义即执行.
# 解决方法一:
# 添加 tf.compat.v1.disable_eager_execution()
# 解决方法二:
# 添加tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
test1_x = tf.compat.v1.placeholder(tf.string)

with tf.compat.v1.Session() as sess:
    output = sess.run(test1_x, feed_dict={test1_x: 'Hello World'})
    print(output)

# test2 : 使用feed_dict参数值设置多个张量
tf.compat.v1.disable_eager_execution()
test2_x = tf.compat.v1.placeholder(tf.string)
test2_y = tf.compat.v1.placeholder(tf.int32, None)
test2_z = tf.compat.v1.placeholder(tf.float32, None)

with tf.compat.v1.Session() as sess:
    output = sess.run(test2_x, feed_dict={test2_x: 'Welcome to CNN', test2_y: 123, test2_z: 123.45})
    print(output)

# test3 : 占位符也可以在多维情况下存储数组
test3_x = tf.compat.v1.placeholder("float", [None, 3])
test3_y = test3_x * 2

with tf.compat.v1.Session() as sess:
    input_data = [[1, 2, 3],
                  [4, 5, 6], ]
    result = sess.run(test3_y, feed_dict={test3_x: input_data})
    print(result)

# test4 : tf.truncated_normal()函数返回一个具有正态分布随机值的张量, 此函数主要用于网络权重初始化
n_features = 5    # 行
n_labels = 2      # 列
weights = tf.compat.v1.truncated_normal((n_features, n_labels))

with tf.compat.v1.Session() as sess:
    print(sess.run(weights))
