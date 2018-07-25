# Author: LIU Zhengying
# Creation Date: 15 June 2018\
"""
Some utilities helping to generate, transform and check TFRecords
in SequenceExample proto (though some functions work for other protos too)
"""

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
  """Generate a TFRecord file in SequenceExample proto from classic matrix
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
    raise ValueError('Features size {} does not match labels size {}.'\
                     .format(num_examples, labels.shape[0]))

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

def checks_exist_and_splits_filename(path_to_file):
  if os.path.exists(path_to_file):
    folder, filename = os.path.split(path_to_file)
    file_wo_ext, ext = os.path.splitext(filename)
    return folder, file_wo_ext, ext
  else:
    raise IOError("The file {} doesn't exist!".format(path_to_file))

def _get_sharded_filenames(path_to_file, num_shards):
  """Create a list of filenames for sharded files.

  Example of resulting files: `<path_to_file>-00001-of-00007.tfrecord`
  """
  folder, file_wo_ext, ext = checks_exist_and_splits_filename(path_to_file)

  if ext == '.tfrecord':
    basename = os.path.join(folder, file_wo_ext)
  else:
    basename = path_to_file

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
    IOError: if cannot find the file.
  Returns:
    filenames: a list of paths of newly generated files.
    Write `num_shards` files named as, say,
      `mnist-train-00001-of-00007.tfrecord` in the same directory
  """

  filenames = _get_sharded_filenames(path_to_tfrecord, num_shards)
  writers = [tf.python_io.TFRecordWriter(x) for x in filenames]

  for i, sequence_example in enumerate(
      tf.python_io.tf_record_iterator(path_to_tfrecord)):
    writer = writers[i % num_shards] # cycle-loop over i
    writer.write(sequence_example)

  [writer.close() for writer in writers]

  if not keep_old_file:
    os.remove(path_to_tfrecord)

  return filenames


def _get_examples_and_labels_filenames(path_to_tfrecord):

  folder, file_wo_ext, ext = checks_exist_and_splits_filename(path_to_tfrecord)

  if ext == '.tfrecord':
    basename = os.path.join(folder, file_wo_ext)
  else:
    basename = path_to_tfrecord

  path_to_examples = basename + "-examples.tfrecord"
  path_to_labels = basename + "-labels.tfrecord"
  return path_to_examples, path_to_labels


def separate_examples_and_labels(path_to_tfrecord, keep_old_file=True):
  """Given a SequenceExample proto containing test data, separates labels from
  examples.

  Args:
    test_data_file: string, path to the file containing test data
  Raises:
    ValueError: if examples in `test_data_file` don't have the `labels` as
      attribute
  Returns:
    Write 2 files with separated examples and labels, both with `id` and return
    the two path string `path_to_examples` and `path_to_labels`.
  """
  path_to_examples, path_to_labels =\
      _get_examples_and_labels_filenames(path_to_tfrecord)

  examples_writer = tf.python_io.TFRecordWriter(path_to_examples)
  labels_writer = tf.python_io.TFRecordWriter(path_to_labels)

  for bytes in tf.python_io.tf_record_iterator(path_to_tfrecord):
    sequence_example = tf.train.SequenceExample.FromString(bytes)

    feature = {}
    for key in ['labels', 'label_index', 'label_score']:
      if key in sequence_example.context.feature:
        value = sequence_example.context.feature.pop(key)
        feature[key] = value

    label_context = tf.train.Features(feature=feature)

    label_sequence_example = tf.train.SequenceExample(
        context=label_context,
        feature_lists={})

    examples_writer.write(sequence_example.SerializeToString())
    labels_writer.write(label_sequence_example.SerializeToString())

  examples_writer.close()
  labels_writer.close()

  if not keep_old_file:
    os.remove(path_to_tfrecord)

  return path_to_examples, path_to_labels


def _get_context_keys(sequence_example):
  return [x for x in sequence_example.context.feature]

def _get_feature_lists_keys(sequence_example):
  return [x for x in sequence_example.feature_lists.feature_list]

def check_file_consistency(path_to_tfrecord):
  """For a given TFRecord, check its consistency. Return its number
  of examples and fields in `context` and `feature_lists`.
  """
  if not os.path.exists(path_to_tfrecord):
    raise ValueError("The path {} doesn't exist!".format(path_to_tfrecord))

  context_keys = []
  feature_lists_keys = []
  num_examples = 0

  for i, bytes in enumerate(tf.python_io.tf_record_iterator(path_to_tfrecord)):

    sequence_example = tf.train.SequenceExample.FromString(bytes)
    context_keys_new = sorted(_get_context_keys(sequence_example))
    feature_lists_keys_new = sorted(_get_feature_lists_keys(sequence_example))

    if context_keys:
      if context_keys != context_keys_new:
        raise ValueError("""Find inconsistent example at index {}.
        Expect context keys {} but got {}."""\
          .format(i, context_keys, context_keys_new))
    else:
      context_keys = context_keys_new

    if feature_lists_keys:
      if feature_lists_keys != feature_lists_keys_new:
        raise ValueError("""Find inconsistent example at index {}.
        Expect context keys {} but got {}."""\
        .format(i, feature_lists_keys, feature_lists_keys_new))
    else:
      feature_lists_keys = feature_lists_keys_new

    num_examples += 1

  # print("""Consistency check done! This TFRecord has {} examples with context {}
  # and feature lists {}"""\
  #   .format(num_examples, context_keys, feature_lists_keys))

  return num_examples, context_keys, feature_lists_keys

def all_identical(li):
  """Given a list `li`, check if all its elements are identical"""
  return li[1:] == li[:-1]

def check_files_consistency(paths_to_tfrecord):
  """Given a list of TFRecords, check its consistency. Return total number
  of examples and fields in `context` and `feature_lists`."""
  check_all = list(map(check_file_consistency, paths_to_tfrecord))
  # print("check_all for {}: {}".format(paths_to_tfrecord, check_all))
  nums_examples = [x for x,_,_ in check_all]
  context_keys_list = [y for _,y,_ in check_all]
  feature_lists_keys_list = [z for _,_,z in check_all]
  assert(all_identical(context_keys_list))
  assert(all_identical(feature_lists_keys_list))
  return sum(nums_examples), context_keys_list[0], feature_lists_keys_list[0]



if __name__ == "__main__":
  import convert_mnist_to_tfrecords as haha
  haha.main() # Download and write MNIST in TFRecords

  # path_to_tfrecord = 'mnist/mnist-test.tfrecord'
  #
  # check_file_consistency(path_to_tfrecord)

  separate_examples_and_labels(path_to_tfrecord, keep_old_file=False)

  shard_tfrecord(path_to_tfrecord='mnist/mnist-test-examples.tfrecord',
                 num_shards=2,
                 keep_old_file=False)
  shard_tfrecord(path_to_tfrecord='mnist/mnist-train.tfrecord',
                 num_shards=12,
                 keep_old_file=False)
