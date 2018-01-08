import gzip
import os
import sys
import urllib
import tensorflow.python.platform
import numpy
import tensorflow as tf

tf.app.flags.DEFINE_string("train_data_filename", "data/train-images-idx3-ubyte.gz", "data of training dataset file")
tf.app.flags.DEFINE_string("train_label_filename", "data/train-labels-idx1-ubyte.gz", "label of training dataset file")
tf.app.flags.DEFINE_string("test_data_filename", "data/t10k-images-idx3-ubyte.gz", "data of test dataset file")
tf.app.flags.DEFINE_string("test_label_filename", "data/t10k-labels-idx1-ubyte.gz", "label of test dataset file")
tf.app.flags.DEFINE_integer("image_size", 28, "size of image")
tf.app.flags.DEFINE_integer("num_channels", 1, "# of channels")
tf.app.flags.DEFINE_integer("pixel_depth", 255, "depth of pixel")
tf.app.flags.DEFINE_integer("num_label", 10, "# of label")
tf.app.flags.DEFINE_integer("validation_size", 5000, "size of validation dataset")
tf.app.flags.DEFINE_integer("seed", 65347, "random seed")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("num_epoch", 5, "# of epoch")
tf.app.flags.DEFINE_float("lr", 0.01, "learning rate")


FLAGS = tf.app.flags.FLAGS

def process_data(data_file, label_file, num_images):
    with gzip.open(data_file) as data_reader:
        data_reader.read(16)
        buf = data_reader.read(FLAGS.image_size * FLAGS.image_size * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (FLAGS.pixel_depth / 2.0)) / FLAGS.pixel_depth
        data = data.reshape(num_images, FLAGS.image_size, FLAGS.image_size, 1)
    with gzip.open(label_file) as label_reader:
        label_reader.read(8)
        buf = label_reader.read(1 * num_images)
        label = numpy.frombuffer(buf, dtype=numpy.uint8)
        label = (numpy.arange(FLAGS.num_label) == label[:, None]).astype(numpy.float32)
    return data, label

def precision_rate(predictions, label):
    num_examples = predictions.shape[0]
    true_counts = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(label, 1))
    prec = float(true_counts)/float(num_examples)
    return prec

def main(argv=None):
    train_data, train_label = process_data(FLAGS.train_data_filename, FLAGS.train_label_filename, 60000)
    test_data, test_label = process_data(FLAGS.test_data_filename, FLAGS.test_label_filename, 10000)

    validation_data = train_data[:FLAGS.validation_size, ...]
    validation_label = train_label[:FLAGS.validation_size]
    train_data = train_data[FLAGS.validation_size:, ...]
    train_label = train_label[FLAGS.validation_size:]

    train_size = train_label.shape[0]

    train_data_feed = tf.placeholder(
        tf.float32,
        shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
    train_label_feed = tf.placeholder(
		tf.float32,
        shape=(FLAGS.batch_size, FLAGS.num_label))
    eval_data = tf.placeholder(
        tf.float32,
        shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))

    conv1_w = tf.Variable(
          tf.truncated_normal([5, 5, FLAGS.num_channels, 32],
                            stddev=0.1,
                            seed=FLAGS.seed))
    conv1_b = tf.Variable(tf.zeros([32]))
    conv2_w = tf.Variable(
          tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=FLAGS.seed))
    conv2_b = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_w = tf.Variable(
          tf.truncated_normal([FLAGS.image_size / 4 * FLAGS.image_size / 4 * 64, 512],
                            stddev=0.1,
                            seed=FLAGS.seed))
    fc1_b = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_w = tf.Variable(
          tf.truncated_normal([512, FLAGS.num_label],
                            stddev=0.1,
                            seed=FLAGS.seed))
    fc2_b = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_label])) 

    def model(data, train=False):
      conv1 = tf.nn.conv2d(data,
                            conv1_w,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
      relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
      pool1 = tf.nn.max_pool(relu1,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
      conv2 = tf.nn.conv2d(pool1,
                            conv2_w,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
      relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
      pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
      pool_shape = pool2.get_shape().as_list()
      reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
      hidden = tf.nn.relu(tf.matmul(reshape, fc1_w) + fc1_b)
      if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=FLAGS.seed)
      return tf.matmul(hidden, fc2_w) + fc2_b
      

    logits = model(train_data_feed, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_label_feed))

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        FLAGS.lr,                
        batch * FLAGS.batch_size,
        train_size,          
        0.95,               
        staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(model(eval_data))

    def eval_in_batches(data, sess):
        size = data.shape[0]
        predictions = numpy.ndarray(shape=(size, FLAGS.num_label), dtype=numpy.float32)
        for begin in xrange(0, size, FLAGS.batch_size):
            end = begin + FLAGS.batch_size
            if end <= size:
                predictions[begin:end, :] = sess.run(
						eval_prediction,
						feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
						eval_prediction,
						feed_dict={eval_data: data[-FLAGS.batch_size:, ...]})
                predictions[begin:, :] = batch_predictions[begin-size:, :]
        return predictions


    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print 'Initialized!'
        for step in xrange(int(FLAGS.num_epoch * train_size / FLAGS.batch_size)):
            epoch = float(step) * FLAGS.batch_size / train_size
            offset = (step * FLAGS.batch_size) % (train_size - FLAGS.batch_size)
            batch_data = train_data[offset:(offset + FLAGS.batch_size), ...]
            batch_label = train_label[offset:(offset + FLAGS.batch_size)]
            feed_dict = {train_data_feed: batch_data,
                         train_label_feed: batch_label}
            _, l, lr, predictions = sess.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            if step % 100 == 0:
                print 'Epoch %d / Read %d' % (epoch, step*FLAGS.batch_size%train_size)
                print 'Batch precision: %.4f%%' % precision_rate(predictions,
                                                             batch_label)
                print 'Validation precision: %.4f%%' % precision_rate(
                    eval_in_batches(validation_data, sess), validation_label)
                sys.stdout.flush()
        test_precision = precision_rate(eval_in_batches(test_data, sess), test_label)
        print 'Test precision: %.04f%%' % test_precision


if __name__ == '__main__':
    tf.app.run()
