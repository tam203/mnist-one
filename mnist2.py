from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None,784])
y_ = tf.placeholder("float", shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
c = tf.Variable(tf.zeros([784,784]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W))

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(50):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


print ("second train...")


y2 = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy2 = -tf.reduce_sum(y_*tf.log(y2))
second_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy2, var_list=[b])
for i in range(1000):
    batch = mnist.train.next_batch(500)
    second_train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction2 = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

print (accuracy2.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#for i in range(1000):
#    batch = mnist.train.next_batch(50)
#    train_step.run(feed_dict={x: batch[0], y_: batch[1]})