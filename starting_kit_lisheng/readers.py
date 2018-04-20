import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import numpy as np
import math
import matplotlib.pyplot as plt
import os


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_sequence_example(
          serialized_example,
          context_features={
              'label_index': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
              'label_score': tf.FixedLenFeature(shape=[1], dtype=tf.float32)},
          sequence_features={
              '0_dense_input': tf.FixedLenSequenceFeature(
                  shape=[28*28], dtype=tf.float32)}) #image in MNIST is of size 28 by 28
  # print ('features: ', features)

  return features[1]['0_dense_input'], features[0]['label_index']


def read_inputs(inputfile_pattern, batch_size=1000): 
  """
  Arguments: 
  - inputfile_pattern: path to the input tfrecord file
  - batch_size: specify how many images the reader will read

  Returns:
  - dense_inputs_flatten: normalized flatten image matrix, of shape (784, batch_size)
  - one_hot_labels: one-hotted labels
  - labels: raw labels

  Raise:
  IOError
  """
  if not os.path.isfile(inputfile_pattern):
    raise IOError("Unable to find training files. inputfile_pattern='" +
                    inputfile_pattern + "'.")
  filename_queue = tf.train.string_input_producer([inputfile_pattern])

  dense_input, label = read_and_decode(filename_queue)
  
  dense_input = tf.reshape(dense_input, [1, 28*28])
  label = tf.reshape(label, [1, ])
  
  _batch_size = batch_size
  dense_inputs, labels = tf.train.shuffle_batch(
          [dense_input, label], batch_size=_batch_size,
          capacity=1 + 3*_batch_size, min_after_dequeue=1)
  dense_inputs_flatten = tf.transpose(tf.squeeze(dense_inputs))

  dense_inputs_flatten /= 255.
  one_hot_labels = tf.transpose(tf.squeeze(tf.one_hot(labels, 10))) 
  return dense_inputs_flatten, one_hot_labels, labels



