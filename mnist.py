from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  im_help
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(
    tf.Print(accuracy,[b], summarize=10),
    feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

WArr = sess.run(W)

newim = im_help.image_from_2d(im_help.reshape(WArr[:,0]), "zero.png")

#print WArr[:,0]
print (WArr[:,1])

print (WArr.shape[1])
for i in range(WArr.shape[1]):
    newim = im_help.image_from_2d(im_help.reshape(WArr[:,i]), "W%s.png"%i)

#tf.Print(b,[b],message="Tensor...")