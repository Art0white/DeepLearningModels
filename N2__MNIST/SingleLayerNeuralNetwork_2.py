import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#使用placeholder
tf.compat.v1.disable_eager_execution()
#设定批次大小
batch_size = 100
#计算批次数量
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.compat.v1.placeholder(tf.float32,[None,784])
y = tf.compat.v1.placeholder(tf.float32,[None,10])

#创建简单的不含隐藏层的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
W_plus_b = tf.matmul(x, W)+b
prediction = tf.nn.softmax(W_plus_b)

#定义二次代价损失函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.2)
#最小化损失函数
train = optimizer.minimize(loss)

#初始化变量
init = tf.compat.v1.global_variables_initializer()

#将结果存到布尔型列表里
accuracy_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) #argmax表示返回一维张量最大值的位置
#计算准确率
accuracy = tf.reduce_mean(tf.cast(accuracy_prediction,tf.float32))     #cast将前者转换为float类型变量

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)      #batch_xs每次获得batch_size大小图片，batch_ys获得标签
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})

        #每一轮输出准确率
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter:'+str(epoch)+' accuracy:'+str(acc))