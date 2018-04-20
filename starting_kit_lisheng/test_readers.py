import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import numpy as np
import math
import matplotlib.pyplot as plt

import readers 
import models 
import utils 
from readers import *





def test():
  # inputfile = 'small-mnist.tfrecord'
  inputfile = 'new_sample.tfrecord'
  with tf.Graph().as_default():
    inputs, one_hot_labels, labels = read_inputs(inputfile)

    init_op = tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)

    coord = tf.train.Coordinator() # The Coordinator class helps multiple threads stop together and report exceptions to a program that waits for them to stop
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) # create threads that use coord
    inputs, labels = sess.run([inputs, labels])
    coord.request_stop()
    coord.join(threads) # waits until the specified threads have stopped
    inputs = sess.run(tf.reshape(inputs, [28, 28, -1])) # 
    for i in xrange(10): # show 10 examples from the 1000 examples returned by read_inputs()
      print ("exmaple %i, label %i"%(i, labels[i]))    
      plt.imshow(inputs[:, :, i], cmap='gray')
      plt.show()
    sess.close()



def test_python_io():
  tfrecords_filename = 'new_sample.tfrecord'
  record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

  # how many examples in the input file?
  i=0
  for string_record in record_iterator:
    example = tf.train.SequenceExample()
    example.ParseFromString(string_record)

    print (example)
    print type(example['feature_lists'])
    break
    i+=1
  print i


if __name__ == '__main__':
  # test_python_io()
  test()


