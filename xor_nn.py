import numpy as np
import tensorflow as tf
import os


# Training params
alpha = 0.01
num_epochs = 10000

# Training data
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])


x = tf.placeholder(tf.float32, [None, 2], name='in_layer')
y = tf.placeholder(tf.float32, [None, 1], name='out_layer')

# Create the graph
W1 = tf.Variable(tf.random_normal([2, 2], name='W1'))
b1 = tf.Variable(tf.random_normal([2], name='b1'))

W2 = tf.Variable(tf.random_normal([2, 1], name='W2'))
b2 = tf.Variable(tf.random_normal([1], name='b2'))

h_out = tf.nn.sigmoid(6.0 * tf.add(tf.matmul(x, W1), b1))
y_ = tf.nn.sigmoid(6.0 * tf.add(tf.matmul(h_out, W2), b2))

# Loss functions and optimizer
mse = tf.reduce_mean((y_ - y) * (y_ - y))
x_entropy = -tf.reduce_mean(y * tf.log(y_) + (1.0 - y) * tf.log(1.0 - y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(x_entropy)

# Values to record for tensorboard
tf.summary.scalar('loss', x_entropy)
tf.summary.scalar('mse', mse)
merged_summaries_op = tf.summary.merge_all()

# Run the training
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.getcwd(), sess.graph_def)
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        _, loss, err, summary = sess.run([optimizer, x_entropy, mse, merged_summaries_op], feed_dict={x: X, y: Y})
        writer.add_summary(summary, global_step=epoch)
        if epoch % (num_epochs // 10) == 0:
            print("epoch: {0}\t-- loss = {1:.5f} \tMSE = {2:.5f}".format(epoch, loss, err))
