# Author: LIU Zhengying
# Creation Date: 15 June 2018
# Description: Some utilities helping to generate and transform TFRecords
#   in SequenceExample proto

import tensorflow as tf
import os

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

def convert_matrix_to_tfrecord(features, labels, dataset_name, mode=None):
  """Generate TFRecords in SequenceExample proto from classic matrix
  representation.

  Args:
    features: A (dense) numpy array of shape (num_examples, num_features).
    labels: A numpy array of shape (num_examples, ).
    dataset_name: A string indicating the dataset name such as `mnist`.
    mode: A string, can be `train`, `valid` or `test`.
  Raises:
    ValueError: If number of examples does not match number of lines of labels
  Returns:
    None: Write a file of TFRecords in the current directory under the name
      `<dataset_name>-<mode>.tfrecord`
  """
  num_examples = features.shape[0]
  if num_examples != labels.shape[0]:
    raise ValueError('Features size %d does not match labels size %d.' %
                     (num_examples, labels.shape[0]))

  if mode:
    filename = dataset_name + '-' + mode + '.tfrecord'
  else:
    filename = dataset_name + '.tfrecord'

  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      context = tf.train.Features(
            feature={
                'id': _int64_feature(index),  # Use index as id
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

def convert_AutoML_to_AutoDL(*arg, **kwarg):
  """Convert a dataset in AutoML format to AutoDL format.

  This facilitates the process of generating new datasets in AutoDL format,
  since there exists a big database of datasets in AutoML format.
  """
  pass

def _get_basename(path_to_file)

def _get_sharded_filenames(path_to_file, num_shards):
  """Create a list of filenames for sharded files.
  
  Example of resulting files: `<path_to_file>-00001-of-00007.tfrecord`
  """
  
  if path_to_file.endswith('.tfrecord'):
    basename = os.path.splitext(path_to_file)[0]  # remove extension
  
  output_filenames = [basename + 
                      "-{:05d}-of-{:05d}.tfrecord".format(i, num_shards)
                      for i in range(num_shards)]
  
  return output_filenames

def shard_tfrecord(path_to_tfrecord, num_shards, keep_old_file=True):
  """Shards one TFRecord file into small pieces in the same format.

  Args:
    path_to_tfrecord: string, path to the TFRecord file.
    num_shards: int, number of resulting TFRecord files. num_shards should be 
      less than 10000.
    keep_old_file: bool, optional. Whether keep old TFRecord file.
  Raises:
    IOError: If cannot find the file
  Returns:
    None: Write `num_shards` files named as, say,
      `mnist-train-00001-of-00007.tfrecord` in the same directory
  """
  
  filenames = _get_sharded_filenames(path_to_tfrecord, num_shards)
  writers = [tf.python_io.TFRecordWriter(x) for x in filenames]
  
  for i,example in enumerate(tf.python_io.tf_record_iterator(path_to_tfrecord)):
    writer = writers[i % num_shards]
    writer.write(example)
  
  [writer.close() for writer in writers]
  
  if not keep_old_file:
    os.remove(path_to_tfrecord)
    
def separate_examples_and_labels(test_data_file):
  """Given a file containing test data, separates labels from examples.
  
  Args:
    test_data_file: string, path to the file containing test data
  Raises:
    ValueError: if examples in `test_data_file` don't have the `labels` as
      attribute
  Returns:
    None: Write 2 files with separated examples and labels, both with `id`
  """
  
  if path_to_file.endswith('.tfrecord'):
    basename = os.path.splitext(path_to_file)[0]  # remove extension
  
  
  
if __name__ == '__main__':
  filename = 'mnist-test.tfrecord'
  shard_tfrecord(filename,2)
  
