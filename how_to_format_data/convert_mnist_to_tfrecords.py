# Author: LIU Zhengying
# Creation Date: 17 April 2018
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
  tf.train.Features and `feature_lists` is of type tf.train.FeatureLists.

  See:
    https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto
  for more information.

  In our case, `context` will be the label and `feature_lists` will be, say, the
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

  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      context = tf.train.Features(
            feature={
                'id': _int64_feature(index), # use index as id
                'label_index': _int64_feature(labels[index]),
                'label_score': _float_feature([1])
            })
      feature_lists = tf.train.FeatureLists(
          feature_list={
          '0_dense_input': _feature_list([_float_feature(features[index])])
          })
      sequence_example = tf.train.SequenceExample(
          context=context,
          feature_lists=feature_lists)
      writer.write(sequence_example.SerializeToString())


def main():
  datasets = mnist.read_data_sets(train_dir='/tmp/data/', validation_size=0)
  print("Training data size:", datasets.train.images.shape)
  print("Validation data size:", datasets.validation.images.shape)
  print("Test data size:", datasets.test.images.shape)

  input_sequence = datasets.train.images
  output_sequence = datasets.train.labels
  filename_train = 'mnist/mnist-train.tfrecord'

  convert_to_sequence_example_tfrecords(
      features=input_sequence,
      labels=output_sequence,
      filename=filename_train)
  print("Conversion for training set is done.")

  input_sequence = datasets.test.images
  output_sequence = datasets.test.labels
  filename_test = 'mnist/mnist-test.tfrecord'

  convert_to_sequence_example_tfrecords(
      features=input_sequence,
      labels=output_sequence,
      filename=filename_test)
  print("Conversion for test set is done.")


  from tfrecord_utils import *
  print("Separating labels from examples for test set...", end='')
  separate_examples_and_labels(path_to_tfrecord, keep_old_file=False)
  print("Done!")

  print("Sharding training set and test set...", end='')
  shard_tfrecord(path_to_tfrecord='mnist/mnist-test-examples.tfrecord',
                 num_shards=2,
                 keep_old_file=False)
  shard_tfrecord(path_to_tfrecord='mnist/mnist-train.tfrecord',
                 num_shards=12,
                 keep_old_file=False)
  print("Done!")

  print("Adding metadata...", end='')
  filename_metadata = 'mnist/metadata.textproto'
  metadata =
  """is_sequence: false
sample_count: 60000
output_dim: 10
matrix_spec {
  col_count: 28
  row_count: 28
  is_sequence_col: false
  is_sequence_row: false
  has_locality_col: true
  has_locality_row: true
  format: DENSE
}
"""
  with open(filename_metadata, 'w') as f:
    f.write(metadata)
  print("Done!")

  print("Finished creating MNIST dataset under TFRecord format. You can find"
        "the dataset in the directory mnist/."
  )


if __name__ == "__main__":
  main()
