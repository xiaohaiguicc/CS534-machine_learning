import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
"""With corrected initialization condition"""

print('It took Qingfeng about 400~700s to finish the program')
start_time = time.time()


# Define simple conv layer
def conv_layer(input, channels_in, channels_out, name='conv'):
  with tf.name_scope(name):
    w = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]), name='Weight')
    b = tf.Variable(tf.zeros([channels_out]), name='Bias')
    conv = tf.nn.conv2d(input, w, strides = [1,1,1,1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram('weights', w)
    tf.summary.histogram('biases',b)
    tf.summary.histogram('activations',act)
    return act

# Define fully connected layer
def fc_layer(input, channels_in, channels_out, name='fc'):
  with tf.name_scope(name):
    w = tf.Variable(tf.zeros([channels_in, channels_out]), name='Weight')
    b = tf.Variable(tf.zeros([channels_out]), name='Bias')
    act = tf.nn.relu(tf.matmul(input, w) + b)
    return act

# Setup placeholders, and reshape the data
x = tf.placeholder(tf.float32, shape = [None, 784], name='x_inputs')
y = tf.placeholder(tf.float32, shape = [None, 10],name='labels')
x_image = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', x_image, max_outputs = 3)


# Create the network
conv1 = conv_layer(x_image, 1, 32, 'conv1') #Channels_out = 32 meaning we want 32 filters
pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
#Only shrinks height and length but not DEPTH, here he depth (channel) is still 32

conv2 = conv_layer(pool1, 32, 64, 'conv2') #Meaning input 32 filters
pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
flattened = tf.reshape(pool2, [-1, 7*7*64]) # 7 = 28/2/2, because 2 pooling layers.
# Here each image is transformed into a 7*7*64 numbers. -1 is to indicate how many images are there.

fc1 = fc_layer(flattened, 7*7*64, 1024, 'fc1') #Each image 7*7*64 are converted to a 1024 node.
logits = fc_layer(fc1, 1024, 10, 'fc2') # Convert 1024 node to 10 nodes

# Compute cross entropy as our loss function
with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
  tf.summary.scalar('cross_entropy', cross_entropy)


# Use an AdamOptimizer to train the network
with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Compute the accuracy
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# argmax(input, axis):Returns the index with the largest value across axes of a tensor

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

# This is another code I think is missing
sess = tf.InteractiveSession()
# Initialize all parameters
sess.run(tf.global_variables_initializer())

#I think this is missing required code for mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Merge all summaries
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("../../tensorboard/youtube_example/3")
writer.add_graph(sess.graph)

# Train for 2000 steps
for i in range(2000):
    batch = mnist.train.next_batch(100)
    if i%5 == 0:
        s = sess.run(merged_summary,feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, i)

        [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
        print("step %d, training accuracy %g"%(i,train_accuracy))

    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})




print("--- %s seconds ---" % (time.time() - start_time))