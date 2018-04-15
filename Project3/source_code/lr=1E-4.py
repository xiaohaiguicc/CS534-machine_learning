import tensorflow as tf
import numpy as np


# Parameters
# Default: list all the default variables that need to tune


# Adjust: put the parameter that need to be adjusted
learning_rate = 1E-4

# Save dir:
save_dir = 'tensorboard/learning_rate/lr=1E-4'




#This is to list all INFO
tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN. This function is used to be called later in main function.
    features    :: the feature in array, size nnumber-by-784. Here 784 = 28*28 is the flattened representation of a hand written figure.
                    Features is a dict structure,  {'x': <tf.Tensor 'fifo_queue_DequeueUpTo:1' shape=(nnumber, 784) dtype=float32>}
                    features['x'] corresponds to the acutal tensor object
    labels      :: values from 0 to 9. Size nnumber-by-1. A tensor object
    mode        :: One of three modes: TRAIN, EVAL, PREDICT
  """

  # It is a function that used to train data.

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu, name='conv1')

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name = 'pool1')

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu, name = 'conv2')

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)



  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),#Calculate class on the fly
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  tf.summary.scalar('loss', loss) #Write the loss into tensorboard


  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)#potimized and learning rate can change
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}#predictions is a dict and calculate classes on the fly
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(aa):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    p = int(len(mnist.train.images)*0.8) #Probably need to randomize
    with tf.name_scope('train_data'):
        train_data = mnist.train.images[0:p] # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)[0:p]

    with tf.name_scope('valid_data'):
        valid_data = mnist.train.images[p:]  # Returns np.array
        valid_labels = np.asarray(mnist.train.labels, dtype=np.int32)[p:]

    with tf.name_scope('test_data'):
        eval_data = mnist.test.images # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=save_dir)
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # # Merge all summaries
    # merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("../../tensorboard/project3/pj3-1")
    # writer.add_graph(sess.graph)


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},#First input, convert the numpy array data into a dict structure
        y=train_labels,
        batch_size= 200,
        num_epochs=None, #Will run forever
        shuffle=True)

    valid_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": valid_data},
        y=valid_labels,
        num_epochs=1,
        shuffle=False) # Boolean, if True shuffles the queue. Avoid shuffle at prediction time.

    experiment = tf.contrib.learn.Experiment(
        mnist_classifier,
        train_input_fn,
        valid_input_fn,
        train_steps = 5000, #This is the step for gradient?
        eval_steps = None,
        train_steps_per_iteration = 500) #Every this step, save to ckpt, and evaluate accuracy
    experiment.continuous_train_and_eval()
    # The result of this step is a trained mnist_classifer

    #The rest come from tutorial
    # mnist_classifier.train(
    #     input_fn=train_input_fn,
    #     steps=1,
    #     hooks=[logging_hook])


    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)




if __name__ == '__main__':
    """Runs whole fitting program automatically"""
    import time

    start_time = time.time()
    tf.app.run()
    print("--- %s seconds ---" % (time.time() - start_time))



