import numpy as np
import tensorflow as tf
import os


# Training params
learning_rate = 0.1
num_epochs = 10000

# Training data
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])

with tf.name_scope('inputs') as scope:
    x = tf.placeholder(tf.float32, [None, 2], name='in_layer')
    y = tf.placeholder(tf.float32, [None, 1], name='output_target')

# Create the graph: hidden layer
with tf.name_scope('hidden') as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name='W1')
    b1 = tf.Variable(tf.random_normal([2]), name='b1')
    h_out = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1), name='hidden_layer')
    with tf.name_scope('summaries') as scope2:
        tf.summary.histogram('weights', W1)
        tf.summary.histogram('biases', b1)
        tf.summary.histogram('activations', h_out)
        tf.summary.scalar('w1', W1[0][0])
        tf.summary.scalar('w2', W1[0][1])
        tf.summary.scalar('w3', W1[1][0])
        tf.summary.scalar('w4', W1[1][1])
        tf.summary.scalar('b1', b1[0])
        tf.summary.scalar('b2', b1[1])

# Create the graph: output layer
with tf.name_scope('output') as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name='W2')
    b2 = tf.Variable(tf.random_normal([1]), name='b2')
    y_ = tf.nn.sigmoid(tf.add(tf.matmul(h_out, W2), b2), name='output_actual')
    with tf.name_scope('summaries') as scope2:
        tf.summary.histogram('weights', W2)
        tf.summary.histogram('biases', b2)
        tf.summary.histogram('activations', y_)
        tf.summary.scalar('w5', W2[0][0])
        tf.summary.scalar('w6', W2[1][0])
        tf.summary.scalar('b3', b2[0])

# Loss functions and optimizer
with tf.name_scope('mse') as scope:
    mse = tf.reduce_mean((y_ - y) * (y_ - y), name='mse')

with tf.name_scope('cross_entropy') as scope:
    cross_entropy = -tf.reduce_mean(y * tf.log(y_) + (1.0 - y) * tf.log(1.0 - y_), name='cross_entropy')

with tf.name_scope('optimizer') as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimization_op = optimizer.minimize(cross_entropy)

# Loss function values to record for tensorboard
tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('mse', mse)

# Merge all summaries together
merged_summaries_op = tf.summary.merge_all()

# Run the training
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.getcwd(), sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        _, loss, mse_err, summary = sess.run([optimization_op, cross_entropy, mse, merged_summaries_op],
                                             feed_dict={x: X, y: Y})
        writer.add_summary(summary, global_step=epoch)
        if epoch % (num_epochs // 10) == 0:
            print("epoch: {0}\t-- loss = {1:.5f} \tMSE = {2:.5f}".format(epoch, loss, mse_err))
