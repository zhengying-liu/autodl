# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""Trivial example of learning algorithm."""

import tensorflow as tf
import algorithm
import numpy as np
import os

class Model(algorithm.Algorithm):
  """A neural network with no hidden layer."""

  def __init__(self, metadata):
    super(Model, self).__init__(metadata)

    # Checkpoints configuration
    self.my_checkpointing_config = tf.estimator.RunConfig(
      save_checkpoints_secs = 10,  # Save checkpoints every 10 seconds.
      keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
    )

    # Get dataset name. To be changed.
    self.dataset_name = self.metadata_.get_dataset_name()\
                          .split('/')[-2].split('.')[0]

    # IMPORTANT: directory to store checkpoints of the model
    self.checkpoints_dir = 'checkpoints_' + self.dataset_name

    # Classifier
    self.classifier = tf.estimator.Estimator(
      model_fn=self.model_fn,
      model_dir=self.checkpoints_dir,
      config=self.my_checkpointing_config)

    self.is_trained = False

  def model_fn(self, features, labels, mode):
    """Model function to construct TensorFlow estimator.

    For details see:
    https://www.tensorflow.org/get_started/custom_estimators#write_a_model_function
    """
    col_count, row_count = self.metadata_.get_matrix_size(0)
    sequence_size = self.metadata_.get_sequence_size()
    output_dim = self.metadata_.get_output_size()

    # Construct a neural network with 0 hidden layer
    input_layer = tf.reshape(features['x'],
                             [-1, sequence_size*row_count*col_count])

    # Replace missing values by 0
    input_layer = tf.where(tf.is_nan(input_layer),
                           tf.zeros_like(input_layer), input_layer)

    logits = tf.layers.dense(inputs=input_layer, units=output_dim)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  def train(self, dataset):
    """Train the model on `dataset`

    Args:
    - dataset: A tf.data.Dataset object
    """

    # Turn `features` in the tensor pair (features, labels) to a dict
    dataset = dataset.map(lambda x, y: ({'x': x}, y))

    # # Set up logging for predictions
    # # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=50)

    def train_input_fn():
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    with tf.Session() as sess:
      self.classifier.train(
        input_fn=train_input_fn,
        steps=2000)#,
        # hooks=[logging_hook])

    self.is_trained = True

  def predict(self, *input_arg):
    """Make prediction for one single example."""
    return self.first_example_output

  def test(self, dataset):
    """
    Given a dataset, make predictions using self.classifier.predict() on
    all examples.

    Args:
    - dataset: A tf.data.Dataset object
    Return:
    - res: A np.ndarray matrix of shape (num_examples, output_dim)
    """
    # Turn `features` in the tensor pair (features, labels) to a dict
    dataset = dataset.map(lambda x, y: ({'x': x}, y))

    def test_input_fn():
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    res = []
    test_results = self.classifier.predict(input_fn=test_input_fn)
    res = [x['probabilities'] for x in test_results]
    res = np.array(res)
    return res
