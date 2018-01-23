import numpy as np
import tensorflow as tf
import os


# Training params
learning_rate = 0.1
num_epochs = 10000
num_prints = 10

# Training data
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])

x = tf.placeholder(tf.float32, [None, 2], name='in_layer')
y = tf.placeholder(tf.float32, [None, 1], name='output_target')

# Create the graph: hidden layer
W1 = tf.Variable(tf.random_normal([2, 2]), name='W1')
b1 = tf.Variable(tf.random_normal([2]), name='b1')
h_out = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1), name='hidden_layer')

# Create the graph: output layer
W2 = tf.Variable(tf.random_normal([2, 1]), name='W2')
b2 = tf.Variable(tf.random_normal([1]), name='b2')
y_ = tf.nn.sigmoid(tf.add(tf.matmul(h_out, W2), b2), name='output_actual')

# Loss functions and optimizer
mse = tf.reduce_mean((y_ - y) * (y_ - y), name='mse')

cross_entropy = -tf.reduce_mean(y * tf.log(y_) + (1.0 - y) * tf.log(1.0 - y_), name='cross_entropy')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimization_op = optimizer.minimize(cross_entropy)

# Run the training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        _, loss, mse_err = sess.run([optimization_op, cross_entropy, mse],
                                    feed_dict={x: X, y: Y})
        if epoch % (num_epochs // num_prints) == 0:
            print("epoch: {0}\t-- loss = {1:.5f} \tMSE = {2:.5f}".format(epoch, loss, mse_err))
