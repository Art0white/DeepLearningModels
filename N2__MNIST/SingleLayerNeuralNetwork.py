import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 下载 MNIST 数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# TensorFlow2.X将datasets集成到keras的高级接口，而且找不到tensorflow_core文件夹，
# 目前没看懂这个方法 ——> 正确如下： import tensorflow as tf mnist = tf.keras.datasets.mnist (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 所以需要去github找到老版本，把里面的examples文件夹搬过来




tf.compat.v1.disable_eager_execution()
# 设置占位符和变量的代码如下
# All the pixels in the image (28 * 28 = 784)
features_count = 784
# there are 10 digits i.e labels
labels_count = 10
batch_size = 128
epochs = 10
learning_rate = 0.5

features = tf.compat.v1.placeholder(tf.float32, [None, features_count])
labels = tf.compat.v1.placeholder(tf.float32, [None, labels_count])

# See the weights and biases tensors
weights = tf.compat.v1.Variable(tf.compat.v1.truncated_normal((features_count, labels_count)))
biases = tf.compat.v1.Variable(tf.compat.v1.zeros(labels_count), name='biases')

# 在开始训练前, 先构建变量初始化计算, 然后构建测量预测准确率的运算, 如下所示:
# Linear Function WX + b
logits = tf.compat.v1.add(tf.compat.v1.matmul(features, weights), biases)
prediction = tf.nn.softmax(logits)

# Cross entropy
# cross_entropy = -tf.reduce_sum(labels * tf.compat.v1.log(prediction), reduction_indices=1) reduction_indices已经被遗弃
cross_entropy = -tf.reduce_sum(labels * tf.compat.v1.log(prediction), axis=1)

# Training loss
loss = tf.reduce_mean(cross_entropy)

# Initializing all variables
init = tf.compat.v1.global_variables_initializer()

# Determining if the predictions are accurate
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

# Calculating prediction accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

# 下面在 TensorFlow 中设置优化器
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 现在开始训练模型, 如以下代码片段所示
# Beginning the session
with tf.compat.v1.Session() as sess:
    # initializing all the variables
    sess.run(init)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={features:batch_x, labels:batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={features:mnist.test.images, labels:mnist.test.labels}))