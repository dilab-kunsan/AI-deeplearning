import tensorflow.python.platform
import numpy as np
import tensorflow as tf


import cPickle as pkl

# flag
dropout=0.5
lr=0.01
input_dim=9894
epoch=10
batch_size=10
num_class = 15
layer = [2000,2000,2000,1000,1000]

def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
		return i+1

def load_data(typ):
	data_path = 'data/'+typ+'_data.txt'
	label_path = 'data/'+typ+'_label.txt'
	f = open(data_path, 'rb')
	numb = file_len(label_path)
	data = np.ndarray((numb, input_dim), dtype=float)
	num_d = 0; idx =0
	for idx, line in enumerate(f):
		b = line.split(' ')
		for idx2, item in enumerate(b):
			if(idx2==input_dim):
				break
			data[idx, idx2]=float(item)
	f.close()
	
	g = open(label_path, 'rb')
	a = np.array([])
	label = np.array([])
	for line in g:
		a = line.split()
		for item in a:
			label = np.append(label, item)
	print '---------------------'
	print 'Data shape	: ', numb, input_dim
	print 'Label shape	: ', numb
	g.close()

	return data, label, numb


def variable_init_2D(num_input, num_output):
  """Initialize weight matrix using truncated normal method
      check detail from Lecun (98) paper.
       - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  """
  init_tensor = tf.truncated_normal([num_input, num_output], stddev=1.0 / math.sqrt(float(num_input)))
  return init_tensor

import math
def build_model(images, input_dim, output_dim, hidden_1=layer[0], hidden_2=layer[1], hidden_3=layer[2], hidden_4=layer[3], hidden_5=layer[4]):#, hidden_6=1000, hidden_7=1000, hidden_8=1000, hidden_9=1000, hidden_10=1000):
  """ Build neural network model
  """
  # X --> Hidden 1
  with tf.name_scope('hidden1') as scope:
    w1      = tf.Variable( variable_init_2D(input_dim, hidden_1), name='w1')
    b1      = tf.Variable( tf.zeros([hidden_1]), name='b2')
    h1 = tf.nn.relu(  tf.matmul(images, w1) + b1)
    out_h1 = tf.nn.dropout(h1, dropout)
  # Hidden 1 --> Hidden 2
  with tf.name_scope('hidden2') as scope:
    w2      = tf.Variable( variable_init_2D(hidden_1, hidden_2), name='w2')
    b2      = tf.Variable( tf.zeros([hidden_2]), name='b2')
    h2 = tf.nn.relu(  tf.matmul(out_h1, w2) + b2)
    out_h2 = tf.nn.dropout(h2, dropout)
  # Hidden 2 --> Hidden 3
  with tf.name_scope('hidden3') as scope:
    w3      = tf.Variable( variable_init_2D(hidden_2, hidden_3), name='w3')
    b3      = tf.Variable( tf.zeros([hidden_3]), name='b3')
    h3 = tf.nn.relu(  tf.matmul(out_h2, w3) + b3)
    out_h3 = tf.nn.dropout(h3, dropout)

  # Hidden 3 --> Hidden 4
  with tf.name_scope('hidden4') as scope:
    w4      = tf.Variable( variable_init_2D(hidden_3, hidden_4), name='w4')
    b4      = tf.Variable( tf.zeros([hidden_4]), name='b4')
    h4 = tf.nn.relu(  tf.matmul(out_h3, w4) + b4)
    out_h4 = tf.nn.dropout(h4, dropout)

  # Hidden 4 --> Hidden 5
  with tf.name_scope('hidden5') as scope:
    w5      = tf.Variable( variable_init_2D(hidden_4, hidden_5), name='w5')
    b5      = tf.Variable( tf.zeros([hidden_5]), name='b5')
    h5 = tf.nn.relu(  tf.matmul(out_h4, w5) + b5)
    out_h5 = tf.nn.dropout(h5, dropout)

  with tf.name_scope('softmax') as scope:
    w6      = tf.Variable( variable_init_2D(hidden_5, output_dim), name='w11')
    b6      = tf.Variable(tf.zeros([output_dim]), name='b11')
    logits = tf.matmul(out_h5, w6) + b6

  return logits


def cross_entropy_loss(logits, labels, num_class):
    batch_size    = tf.size(labels)
    labels        = tf.expand_dims(labels, 1)
    indices       = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated      = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, num_class]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def next_batch(images, labels, batch_size, numb):
  start  = 0
  num_ex = numb

  while True:
    end = start + batch_size
    if end > num_ex: break
    yield images[start:end], labels[start:end]
    start = end

import numpy as np
def check_performance(sess, eval_op, pl_images, pl_labels, images, labels, batch_size, numb):
  """ check precision performance """
  true_counts = []
  num_examples = 0
  for batch_images, batch_labels in next_batch(images, labels, batch_size, numb):
    feed_dict = {
        pl_images: batch_images,
        pl_labels: batch_labels,
    }
    true_count = sess.run(eval_op, feed_dict=feed_dict)
    true_counts.append( true_count )
    num_examples += batch_size

  return num_examples, true_counts


import sys
def stop_here(): sys.exit()

def main(_):
  images, labels, numb = load_data('train')

  OUTPUT_DIM = num_class
  with tf.Graph().as_default():

    pl_images = tf.placeholder(tf.float32, shape=(batch_size, input_dim), name='pl_image')
    pl_labels = tf.placeholder(tf.int32,   shape=(batch_size), name='pl_label')

    # Model Build - Block 1
    logits = build_model(pl_images, input_dim, OUTPUT_DIM)

    # Loss for update - Block 2
    loss = cross_entropy_loss(logits, pl_labels, num_class)

    # Parameter Update operator - Block 3
    adam = tf.Variable(float(lr), trainable=False)
    optimizer = tf.train.AdamOptimizer(adam)

    train_op = optimizer.minimize(loss)


    # Evaluation Operator
    num_corrects = tf.nn.in_top_k(logits, pl_labels, 1)
    eval_op      = tf.reduce_sum(tf.cast(num_corrects, tf.int32)) # how many

    # Session
    sess = tf.Session()


    # Init all variable
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess.run(init)

    # --------------------------------------------------------------------#
    # train - Block 4
    step = 0
    check_losses = []
    previous_losses = []
    prec = 0
    for ep in range(epoch):
      for idx, (batch_images, batch_labels) in enumerate(next_batch(images, labels, batch_size, numb)):
        feed_dict = {
            pl_images: batch_images,
            pl_labels: batch_labels,
        }

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        check_losses.append( loss_value )

        """
        if prec > 0.6 and train_op.name != 'Adam':
          sess.run(learning_rate_adam) # reset learning rate for new optimizer
          train_op = adam_op  # change optimizer
          previous_losses.append( check_mean_loss ) # reset history
        """
        '''
        if idx % print_idx == 0:
          b_num_examples, b_true_counts = check_performance(sess, eval_op, pl_images, pl_labels, batch_images, batch_labels, batch_size, batch_size)
          all_num_examples  =  b_num_examples
          total_true_counts = np.sum(b_true_counts)
          prec              = float(total_true_counts) / float(all_num_examples)
          print '\tBatch%d\tNum examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (idx, all_num_examples, total_true_counts, prec )
        '''
      if True:
        b_num_examples, b_true_counts = check_performance(sess, eval_op, pl_images, pl_labels, images, labels, batch_size, numb)
        all_num_examples  =  b_num_examples
        total_true_counts = np.sum(b_true_counts)
        prec              = float(total_true_counts) / float(all_num_examples)
        print ep, 'Train\tNum examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (all_num_examples, total_true_counts, prec )
  
  images2, labels2, num_examples2 = load_data('test')
  true_count=0; idx=0
  for b_i2, b_l2 in next_batch(images2, labels2, batch_size, num_examples2):
	  f_d2 = {pl_images: b_i2, pl_labels: b_l2,}
	  true_count += sess.run(num_corrects, feed_dict=f_d2)
  t_t = np.sum(true_count)
  prec2 = float(t_t)/float(num_examples2)
  print  'Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples2, t_t, prec2 )

if __name__ == '__main__':
    tf.app.run()
