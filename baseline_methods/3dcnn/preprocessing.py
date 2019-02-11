# Author: Zhengying LIU
# Date: 5 Feb 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess 3D tensors (gray-scale images, gray-scale
videos, speech/time series, text, etc).

If the example size is not fixed (e.g. images of different size), crop a region
then rescale to a fixed size with fixed height-width ratio.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import time

# tf.enable_eager_execution()

_GLOBAL_CROP_SIZE = (224,224)
_GLOBAL_NUM_FRAMES = 10
_GLOBAL_NUM_REPEAT = 4
_GLOBAL_CROP_RATIO = 0.5
_SHUFFLE_BUFFER = 1000

def _crop_3d_tensor(tensor_3d,
                    crop_size=_GLOBAL_CROP_SIZE,
                    num_frames=_GLOBAL_NUM_FRAMES,
                    num_repeat=_GLOBAL_NUM_REPEAT,
                    crop_ratio=_GLOBAL_CROP_RATIO):
  """Crop a batch of 3-D tensors to `crop_size`.

  Args:
    tensor_3d: A 3-D tensor batch of shape
      (batch_size, sequence_size, row_count, col_count)
    crop_size: A Tensor of type `int32`. A 1-D tensor of 2 elements,
      size = [crop_height, crop_width]. All cropped image patches are
      resized to this size. The aspect ratio of the image content is not
      preserved. Both crop_height and crop_width need to be positive.
    num_frames: Number of frames to keep (crop).
    num_repeat: The number of repetition of cropping cycle for each batch.
    crop_ratio: The ratio when cropping height and width.
  Returns:
    A Tensor of shape
      [num_repeat * batch_size, num_frames, crop_height, crop_width]
      where crop_size is equal to (crop_height, crop_width).
  """
  if not tensor_3d.shape.ndims == 4:
    raise ValueError("The shape of the tensor to crop should be " +
                     "[batch_size, sequence_size, row_count, col_count]!")
  batch_size, sequence_size, row_count, col_count = tensor_3d.shape
  # Crop time axis
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_3d)[1], 0)
  padded_tensor = tf.pad(tensor_3d, ((0,0), (0, pad_size), (0, 0), (0, 0)))
  maxval = padded_tensor.shape[1] - num_frames + 1
  # Randomly choose the beginning index of frames
  begin = np.random.randint(0, maxval)
  sliced_tensor = tf.slice(padded_tensor,
                           begin=[0, begin, 0, 0],
                           size=[-1, num_frames, -1, -1])
  # Crop spatial axes
  # First, transpose from [batch_size, sequence_size, row_count, col_count]
  # to [batch_size, row_count, col_count, sequence_size]
  sliced_tensor = tf.transpose(sliced_tensor, perm=[0, 2, 3, 1])
  # sliced_tensor = tf.transpose(padded_tensor, perm=[0, 2, 3, 1])
  # Then apply `tf.image.crop_and_resize` by precompute some size info
  y1, x1 = tf.random.uniform(shape=[2, num_repeat * batch_size],
                             minval=0,
                             maxval=1 - crop_ratio)
  y2 = y1 + crop_ratio
   # = tf.random.uniform(shape=[num_repeat * batch_size],
   #                       minval=0,
   #                       maxval=1 - crop_ratio)
  x2 = x1 + crop_ratio
  boxes = tf.transpose([y1, x1, y2, x2])
  box_ind = list(range(batch_size)) * num_repeat
  # At last, crop and resize
  resized_tensor = tf.image.crop_and_resize(sliced_tensor,
                                            boxes,
                                            box_ind,
                                            crop_size)
  return tf.transpose(resized_tensor, perm=[0, 3, 1, 2])

def crop_time_axis(tensor_3d, num_frames, begin_index=None):
  """Given a 3-D tensor, take a slice of length `num_frames` on its time axis.

  Args:
    tensor_3d: A Tensor of shape [sequence_size, row_count, col_count]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_3d`.
  """
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_3d)[1], 0)
  padded_tensor = tf.pad(tensor_3d, ((0, pad_size), (0, 0), (0, 0)))

  # If not given, randomly choose the beginning index of frames
  if not begin_index:
    maxval = tf.shape(padded_tensor)[1] - num_frames + 1
    begin_index = tf.random.uniform([1],
                                    minval=0,
                                    maxval=maxval,
                                    dtype=tf.int32)
    begin_index = tf.stack([begin_index[0], 0, 0], name='begin_index')

  sliced_tensor = tf.slice(padded_tensor,
                           begin=begin_index,
                           size=[num_frames, -1, -1])

  return sliced_tensor

def resize_space_axes(tensor_3d, new_row_count, new_col_count):
  """Given a 3-D tensor, resize space axes have have target size.

  Args:
    tensor_3d: A Tensor of shape [sequence_size, row_count, col_count].
    new_row_count: An integer indicating the target row count.
    new_col_count: An integer indicating the target column count.
  Returns:
    A Tensor of shape [sequence_size, target_row_count, target_col_count].
  """
  transposed = tf.transpose(tensor_3d, perm=[1, 2, 0])
  resized = tf.image.resize_images(transposed,
                                   (new_row_count, new_col_count))
  return tf.transpose(resized, perm=[2, 0, 1])

def preprocess_tensor_3d(tensor_3d,
                         input_shape=None,
                         output_shape=None):
  """Preprocess a 3-D tensor.

  Args:
    tensor_3d: A Tensor of shape [sequence_size, row_count, col_count].
    input_shape: The shape [sequence_size, row_count, col_count] of the input
      examples
    output_shape: The shape [sequence_size, row_count, col_count] of the oputput
      examples. All components should be positive.
  """
  if input_shape:
    shape = [x if x > 0 else None for x in input_shape]
    tensor_3d.set_shape(input_shape)
  else:
    tensor_3d.set_shape([None, None, None])
  if output_shape and output_shape[0] > 0:
    num_frames = output_shape[0]
  else:
    num_frames = _GLOBAL_NUM_FRAMES
  if output_shape and output_shape[1] > 0:
    new_row_count = output_shape[1]
  else:
    new_row_count=_GLOBAL_CROP_SIZE[0]
  if output_shape and output_shape[2] > 0:
    new_col_count = output_shape[2]
  else:
    new_col_count=_GLOBAL_CROP_SIZE[1]

  tensor_t = crop_time_axis(tensor_3d, num_frames=num_frames)
  tensor_ts = resize_space_axes(tensor_t,
                                new_row_count=new_row_count,
                                new_col_count=new_col_count)
  return tensor_ts

def parse_record_fn(value, is_training, dtype):
  """For a (features, labels) pair `value`, apply preprocessing.
  """
  # Retrieve first matrix bundle of `features` in the tensor tuples
  #   (matrix_bundle_0,...,matrix_bundle_(N-1), labels)
  # i.e. matrix_bundle_0
  tensor_3d = value[0]
  # Label is the last element of value
  labels = value[-1]
  tensor_3d_preprocessed = preprocess_tensor_3d(tensor_3d)
  print("tensor_3d_preprocessed:", tensor_3d_preprocessed) # TODO
  return tensor_3d_preprocessed, labels

def input_function(dataset,
                   is_training,
                   batch_size,
                   shuffle_buffer=_SHUFFLE_BUFFER,
                   parse_record_fn=parse_record_fn,
                   num_epochs=1,
                   dtype=tf.float32,
                   datasets_num_private_threads=None,
                   num_parallel_batches=1):
  """Given a Dataset of 3-D tensors, return an iterator over the records.

  Inspired from:
    https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py#L49

  Args:
    dataset: A Dataset representing 3-D tensors. Each example in this dataset
      has shape [sequence_size, row_count, col_count].
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of examples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (features, labels) pair.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private
      threadpool created for all datasets computation.
    num_parallel_batches: Number of parallel batches for tf.data.

  Returns:
    Dataset of (features, labels) pairs ready for iteration, where `features` is
      a 4-D tensor with known shape:
      [batch_size, new_sequence_size, new_row_count, new_col_count]
  """

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  # dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Repeats the dataset for the number of epochs to train.
  # dataset = dataset.repeat(num_epochs)

  # Parses the raw records into images and labels.
  dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
          lambda *value: parse_record_fn(value, is_training, dtype),
          batch_size=batch_size,
          num_parallel_batches=num_parallel_batches,
          drop_remainder=False))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  # dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    tf.compat.v1.logging.info('datasets_num_private_threads: %s',
                              datasets_num_private_threads)
    dataset = threadpool.override_threadpool(
        dataset,
        threadpool.PrivateThreadPool(
            datasets_num_private_threads,
            display_name='input_pipeline_thread_pool'))

  return dataset

def print_first_element(dataset):
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()
  writer = tf.summary.FileWriter('.')
  writer.add_graph(tf.get_default_graph())
  writer.flush()
  with tf.Session() as sess:
    sess.run(iterator.initializer)
    show_all_nodes() # TODO: to delete
    haha = sess.run(next_element)
    print(haha)

def test_crop():
  t_shape = (3, 100, 4, 4)
  tensor_3d = tf.random.uniform(t_shape)
  # print("Original tensor:" , tensor_3d, '\n', tensor_3d.shape)
  crop_size = (224, 224)
  cropped_tensor = _crop_3d_tensor(tensor_3d, crop_size)
  print("Cropped tensor:", cropped_tensor, '\n', cropped_tensor.shape)

def test_resize_space_axes():
  t_shape = [None, None, None]
  tensor_3d = tf.placeholder(tf.float32, shape=t_shape)
  print("tensor_3d.shape.eval():", tensor_3d.shape)
  res = resize_space_axes(tensor_3d,
                          new_row_count=_GLOBAL_CROP_SIZE[0],
                          new_col_count=_GLOBAL_CROP_SIZE[1])
  with tf.Session() as sess:
    rand_array = np.random.rand(100, 224, 224)
    print(sess.run(res, feed_dict={tensor_3d: rand_array}))
  print(res.shape)

def test_crop_time_axis():
  t_shape = [None, None, None]
  tensor_3d = tf.placeholder(tf.float32, shape=t_shape)
  print("tensor_3d.shape.eval():", tensor_3d.shape)
  res = crop_time_axis(tensor_3d,
                       num_frames=_GLOBAL_NUM_FRAMES)
  with tf.Session() as sess:
    rand_array = np.random.rand(100, 224, 224)
    # print(sess.run(res, feed_dict={tensor_3d: rand_array}))
    haha = tf.get_default_graph().get_tensor_by_name("begin_stacked:0")
    print(haha)
    print(sess.run(haha, feed_dict={tensor_3d: rand_array}))
  print(res.shape)

def test_tensorflow():
  t_shape = [None, None, None]
  tensor_unknown = tf.placeholder(tf.float32, shape=t_shape)
  # u_shape = [-1, -1, -1]
  # tensor_unknown.set_shape(t_shape)
  print("tensor_unknown:", tensor_unknown)

def test_input_fn():
  """Test for the funtion `input_fn`."""
  # dataset_dir = '/Users/evariste/projects/autodl-contrib/formatted_datasets/itwas/itwas.data/train'
  dataset_dir = '/Users/evariste/projects/autodl-contrib/formatted_datasets/chao/chao.data/train'
  # dataset_dir = '/Users/evariste/projects/autodl-contrib/formatted_datasets/katze/katze.data/train'
  autodl_dataset = AutoDLDataset(dataset_dir)
  dataset = autodl_dataset.get_dataset()
  print_first_element(dataset)
  row_count, col_count  = autodl_dataset.get_metadata().get_matrix_size(0)
  sequence_size = autodl_dataset.get_metadata().get_sequence_size()
  output_dim = autodl_dataset.get_metadata().get_output_size()
  input_size = (sequence_size, row_count, col_count)

  begin_time = time.time()
  transformed_dataset = input_function(dataset,
                                       is_training=True,
                                       batch_size=30,
                                       shuffle_buffer=_SHUFFLE_BUFFER,
                                       parse_record_fn=parse_record_fn,
                                       num_epochs=42,
                                       dtype=tf.float32)
  end_time = time.time()
  print("Transformation time used:", end_time - begin_time)
  print("transformed_dataset:", transformed_dataset)

  print_first_element(transformed_dataset)

def show_all_nodes():
  print("Nodes names:", [n.name for n in tf.get_default_graph().as_graph_def().node])

if __name__ == '__main__':
  import sys
  sys.path.append('/Users/evariste/projects/autodl/codalab_competition_bundle/AutoDL_starting_kit/AutoDL_ingestion_program')
  from dataset import AutoDLDataset
  test_input_fn()
  # test_tensorflow()
  # test_crop_time_axis()
