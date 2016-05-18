import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
num_data = 100

with tf.Graph().as_default():
    x_placeholder = tf.placeholder(tf.float32, shape = (num_data,))
    y_placeholder = tf.placeholder(tf.float32, shape = (num_data,))

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
    W = weight_variable([1])
    b = bias_variable([1])
    y = W * x_placeholder + b

    def variable_summaries(var, name):
      with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
# Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y - y_placeholder))
    tf.scalar_summary('loss_mse', loss)
    variable_summaries(W, 'W')
    variable_summaries(b, 'b')
    summary_op = tf.merge_all_summaries()
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
# Launch the graph.
        sess.run(init)
        summary_writer = tf.train.SummaryWriter('.', sess.graph)
        summary_writer.flush()

# Fit the line.
        epochs = 201
        for step in xrange(epochs):
            sess.run(train, feed_dict={x_placeholder:x_data, y_placeholder:y_data})

# visualize the status
            summary_str = sess.run(summary_op, feed_dict={x_placeholder:x_data, y_placeholder:y_data})
            summary_writer.add_summary(summary_str, step)

            if step % 20 == 0:
                print(step, sess.run(W), sess.run(b))

# evaluate the model
        correct_prediction = tf.less(y-y_placeholder, 10E-5)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print (accuracy.eval(feed_dict={x_placeholder:x_data, y_placeholder:y_data}))
# Learns best fit is W: [0.1], b: [0.3]

# checkpoint saver
        saver = tf.train.Saver()
        saver.save(sess, '.', global_step=step)
