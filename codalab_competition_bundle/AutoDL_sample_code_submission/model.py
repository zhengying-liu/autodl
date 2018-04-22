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

"""Trivial example of learning algorithm."""

import tensorflow as tf
import algorithm
import numpy as np


class Model(algorithm.Algorithm):
  """A neural network with no hidden layer."""

  def __init__(self, metadata):
    super(Model, self).__init__(metadata)
    self.classifier = tf.estimator.Estimator(model_fn=self.model_fn)
    self.is_trained = False

  def model_fn(self, features, labels, mode):
    # To be changed for general datasets
    # They can be got from `self.metadata_`
    # Now these values only work for MNIST
    num_epochs = 10
    col_count = 28
    row_count = 28
    sequence_size = 1
    output_dim = 10
    batch_size = 30

    # Construct a neural network with 0 hidden layer
    input_layer = tf.reshape(features['x'],
                             [-1, sequence_size*row_count*col_count])
    print("#"*50, "model_fn", mode, input_layer.shape)
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
      print("@"*50, "Begin training!!!")
      self.classifier.train(
        input_fn=train_input_fn,
        steps=2000)#,
        # hooks=[logging_hook])
      print("@"*50, "Training finished.")

    self.is_trained = True

    # dataset_iterator = dataset.make_one_shot_iterator()
    # # The next lines assume that
    # # (a) get_next() returns a minibatch of examples
    # # (b) each minibatch is a pair (inputs, outputs)
    # # (c) the outputs has the same length as the inputs
    # # We get the first minibatch by get_next,
    # # then the output by [1], then the first example by [0].
    # with tf.Session() as sess:
    #   self.first_example_output = sess.run(dataset_iterator.get_next()[1][0])
    #   # print("*"*50, "constant.py", dataset_iterator.get_next()[1][0].shape)

  def predict(self, *input_arg):
    return self.first_example_output


  def test(self, dataset):
    """
    Given a dataset, make predictions using self.predict() on all examples.
    TODO: Test this algorithm on the tensorflow |dataset|.

    Args:
    - dataset: A tf.data.Dataset object
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

    # dataset_iterator = dataset.make_one_shot_iterator()
    # Y_test = []
    # with tf.Session() as sess:
    #   # TODO: to redo
    #   while True:
    #     try:
    #       batch = sess.run(dataset_iterator.get_next()[1])
    #       Y_test.append(batch)
    #       print("Model::test","*"*50, len(Y_test) , batch.shape)
    #     except tf.errors.OutOfRangeError:
    #       break
    # res = np.concatenate(Y_test)
    # print("Model::test", res.shape)
    # return res
