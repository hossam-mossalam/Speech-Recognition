import numpy as np
import tensorflow as tf

from speech_utils import *

class SpeechRecognizer:

  def __init__(self, seq_len, freq, classes, cell_sz, lr):
    self.seq_len = seq_len
    self.freq = freq
    self.classes = classes
    self.cell_sz = cell_sz
    self.lr = lr
    self._create_graph()

  def _create_placeholders(self):
    with tf.name_scope('placeholders'):
      self.X = tf.placeholder(tf.float32, [None, self.seq_len, self.freq], 'X')
      self.y = tf.placeholder(tf.int32, [None,], 'y')

  def _create_rnn_cell(self):
    with tf.name_scope('rnn_cell'):
      self.cell = tf.nn.rnn_cell.LSTMCell(self.cell_sz)

  def _create_output_weights(self):
    with tf.name_scope('output_weights'):
      self.W = tf.get_variable('W',
                  [self.cell_sz, self.classes],
                  initializer = tf.contrib.layers.xavier_initializer())

      self.b = tf.get_variable('b', [self.classes],
                  initializer = tf.zeros_initializer)

  def _compute_output(self):
    with tf.name_scope('cell_output'):
      (cell_output, self.cell_state) = tf.nn.dynamic_rnn(self.cell, self.X,
                                                         dtype=tf.float32)

    cell_output = cell_output[:,-1,:]

    self.logits = tf.matmul(cell_output, self.W) + self.b

  def _compute_loss(self):
    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=self.y,
                            logits=self.logits),
                        name = 'loss')

  def _compute_regularization_loss(self):
    pass

  def _compute_accuracy(self):
    if not hasattr(self, 'logits'):
      raise AttributeError
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(self.y,
                                    tf.argmax(self.logits, axis=1,
                                              output_type=tf.int32))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def _create_optimizer(self):
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

  def _create_graph(self):
    self._create_placeholders()
    self._create_rnn_cell()
    self._create_output_weights()
    self._compute_output()
    self._compute_loss()
    self._compute_regularization_loss()
    self._create_optimizer()
    self._compute_accuracy()

if __name__ == '__main__':
  train_X, train_y, val_X, val_y, word2id, id2word = load_data('./data/')

  val_X = val_X[:200]
  val_y = val_y[:200]
  print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)

  for k, v in id2word.items():
    print(k, v)

  sr = SpeechRecognizer(99,161,len(word2id.keys()),512,5e-6)

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    average_loss = 0

    for r in range(10001):
      current_X, current_y = get_mini_batch(train_X, train_y)

      _, loss = sess.run([sr.train_op, sr.loss],
                         feed_dict={sr.X:current_X, sr.y:current_y})

      average_loss += loss

      if r % 1000 == 0:
        if not r == 0:
          average_loss /= 1000
        print('average loss: %.3f' % (average_loss))
        average_loss = 0
        loss, acc = sess.run([sr.loss, sr.accuracy],
                             feed_dict={sr.X:val_X, sr.y:val_y})
        print('validation loss: %.3f accuracy: %.2f' % (loss, acc))

