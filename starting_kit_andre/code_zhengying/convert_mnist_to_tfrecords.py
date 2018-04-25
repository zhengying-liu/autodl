# Author: Zhengying Liu
# Date: 17 April 2018
# Description: Download and convert MNIST (training) dataset to TFRecords
#   according SequenceExample protocol buffer (tf.train.SequenceExample).
#   The code is partly inspired by:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

def _int64_feature(value):
  """Helper function to create a tf.train.Feature conveniently."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  # Here `value` is a list of floats
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _feature_list(feature):
  # Here `feature` is a list of tf.train.Feature
  return tf.train.FeatureList(feature=feature)

def convert_to_sequence_example_tfrecords(features, labels, filename):
  """Convert NumPy arrays `features` and `labels` to SequenceExample proto.
  A SequenceExample proto consists of a series of sequence examples. Each
  sequence example is a pair (context, feature_list), where `context` is of type
  tf.train.Features and `feature_list` is of type tf.train.FeatureListsself.

  See:
    https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto
  for more information.

  In our case, `context` will be the label and `feature_list` will be, say, the
  image (or audio, video, text, etc).

  Args:
  - features: numpy array of shape (num_examples, num_features)
  - labels: sparse label, numpy array of integers of shape (num_examples,)

  Return:
  - Write a file of TFRecords in the current directory under the name `filename`
  """
  num_examples = features.shape[0]
  if num_examples != labels.shape[0]:
    raise ValueError('Features size %d does not match labels size %d.' %
                     (num_examples, labels.shape[0]))

  num_examples = 10 # To be commented
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      context = tf.train.Features(
            feature={
                'label_index': _int64_feature(labels[index]),
                'label_score': _float_feature([0])
            })
      feature_lists = tf.train.FeatureLists(
          feature_list={
          '0_dense_input': _feature_list([_float_feature(features[index])])
          })
      sequence_example = tf.train.SequenceExample(
          context=context,
          feature_lists=feature_lists)
      writer.write(sequence_example.SerializeToString())


if __name__ == "__main__":
  datasets = mnist.read_data_sets(train_dir='/tmp/data/', validation_size=0)
  print("Training data size:", datasets.train.images.shape)
  print("Validation data size:", datasets.validation.images.shape)
  print("Test data size:", datasets.test.images.shape)

  mode = "test" # train or test
  if mode == "train":
    input_sequence = datasets.train.images
    output_sequence = datasets.train.labels
  else:
    input_sequence = datasets.test.images
    output_sequence = datasets.test.labels

  filename = 'sample-00000-of-00001-' + mode

  convert_to_sequence_example_tfrecords(
      features=input_sequence,
      labels=output_sequence,
      filename=filename)
  print("Conversion done! Now you can read %s using Andre's dataset.py."\
        % filename)
