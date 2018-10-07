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

"""A baseline method for the AutoDL challenge.

It implements the 3 methods (i.e. __init__, train, test) whose abstract
definition can be found in algorithm.py, which specifies the competition
protocol in the comments.

It is STRONGLY RECOMMENDED to closely look at this script algorithm.py in the
ingestion program folder (AutoDL_ingestion_program/) since any submission will
be consisted of a model.py script implementing a concrete class of
    algorithm.Algorithm
The submission can be done by uploading a zip file with model.py and an empty
file called `metadata`, like what we have in the folder
AutoDL_sample_code_submission/. And we also provide the option that a submission
can be a zip file zipping all CONTENT (i.e. not the directory) of
AutoDL_starting_kit/.
"""

import tensorflow as tf
import os

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

# Utility packages
import time
import datetime
import numpy as np
np.random.seed(42)

class Model(algorithm.Algorithm):
  """A neural network with no hidden layer.

  Remarks:
    1. Adam Optimizer with default setting is used as optimizer;
    2. No validation set is used to control number of steps to train;
    3. After each call of self.train, update self.cumulated_num_steps and an
        estimated time per step;
    4. Number of steps to train for each call of self.train is chosen randomly
        according to remaining time budget and estimated time per step;
    5. Make all-zero prediction at beginning.
  """
  def __init__(self, metadata):
    super(Model, self).__init__(metadata)

    # Checkpoints configuration
    self.my_checkpointing_config = tf.estimator.RunConfig(
      save_checkpoints_secs = 600,  # Save checkpoints every 10 minutes.
      keep_checkpoint_max = 5,       # Retain the 5 most recent checkpoints.
    )

    # Get dataset name. To be changed.
    self.dataset_name = self.metadata_.get_dataset_name()\
                          .split('/')[-2].split('.')[0]

    # IMPORTANT: directory to store checkpoints of the model
    self.checkpoints_dir = 'checkpoints_' + self.dataset_name
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.checkpoints_dir = os.path.join(current_dir, os.pardir,
                                        self.checkpoints_dir)

    # Classifier using model_fn
    self.classifier = tf.estimator.Estimator(
      model_fn=self.model_fn,
      model_dir=self.checkpoints_dir,
      config=self.my_checkpointing_config)

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.total_train_time = 0
    self.cumulated_num_steps = 0
    self.estimated_time_per_step = None
    self.total_test_time = 0
    self.cumulated_num_tests = 0
    self.estimated_time_test = None
    self.done_training = False

  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    Args:
      dataset: a `tf.data.Dataset` object. Each example is of the form
            (matrix_bundle_0, matrix_bundle_1, ..., matrix_bundle_(N-1), labels)
          where each matrix bundle is a tf.Tensor of shape
            (batch_size, sequence_size, row_count, col_count)
          with default `batch_size`=30 (if you wish you can unbatch and have any
          batch size you want). `labels` is a tf.Tensor of shape
            (batch_size, output_dim)
          The variable `output_dim` represents number of classes of this
          multilabel classification task. For the first version of AutoDL
          challenge, the number of bundles `N` will be set to 1.

      remaining_time_budget: a `float`, the name should be clear. If not `None`,
          this `train` method should terminate within this time budget.
          Otherwise the submission will fail.
    """
    if self.done_training:
      return

    # Turn `features` in the tensor tuples (matrix_bundle_0,...,matrix_bundle_(N-1), labels)
    # to a dict. This example model only uses the first matrix bundle
    # (i.e. matrix_bundle_0)
    dataset = dataset.map(lambda *x: ({'x': x[0]}, x[-1]))

    def train_input_fn():
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    if not remaining_time_budget:
      remaining_time_budget = 1200 # if no time limit is given, set to 20min

    # The following snippet of code intends to do
    # 1. If no training is done before, train for 1 step (one batch);
    # 2. Otherwise, estimate training time per step and time needed for test,
    #    then compare to remaining time budget to compute a potential maximum
    #    number of steps (max_steps) that can be trained within time budget;
    # 3. Choose a number (steps_to_train) randomly between 0 and max_steps / 2
    #    and train for this many steps.
    if not self.estimated_time_per_step:
      steps_to_train = 1
    else:
      if self.estimated_time_test:
        tentative_estimated_time_test = self.estimated_time_test
      else:
        tentative_estimated_time_test = 50 # conservative estimation for test
      max_steps = int((remaining_time_budget - tentative_estimated_time_test) / self.estimated_time_per_step)
      max_steps /= 2 # Halve it to make more predictions
      max_steps = max(max_steps, 0)
      # Choose random number of steps < max_steps for training
      steps_to_train = np.random.randint(0, max_steps + 1)
      # steps_to_train = 100
    if steps_to_train <= 0:
      print_log("Not enough time remaining for training. "
            f"Estimated time for training per step: {self.estimated_time_per_step:.2f}, "
            f"and for test: {tentative_estimated_time_test}, "
            f"but remaining time budget is: {remaining_time_budget:.2f}. "
            "Skipping...")
      self.done_training = True
    else:
      msg_est = ""
      if self.estimated_time_per_step:
        msg_est = f"estimated time for this: " +\
                  f"{steps_to_train * self.estimated_time_per_step:.2f} sec."
      print_log(f"Begin training for another {steps_to_train} steps...",msg_est)
      train_start = time.time()
      # Start training
      with tf.Session() as sess:
        self.classifier.train(
          input_fn=train_input_fn,
          steps=steps_to_train)
      train_end = time.time()
      # Update for time budget managing
      train_duration = train_end - train_start
      self.total_train_time += train_duration
      self.cumulated_num_steps += steps_to_train
      self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
      print_log(f"{steps_to_train} steps trained. {train_duration:.2f} sec used. "
            f"Now total steps trained: {self.cumulated_num_steps}. "
            f"Total time used for training: {self.total_train_time:.2f} sec. "
            f"Current estimated time per step: {self.estimated_time_per_step:.2e} sec.")

  def test(self, dataset, remaining_time_budget=None):
    """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set.
    """
    if self.done_training:
      return None

    # Turn `features` in the tensor pair (features, labels) to a dict
    dataset = dataset.map(lambda *x: ({'x': x[0]}, x[-1]))

    def test_input_fn():
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    # The following snippet of code intends to do:
    # 0. With probability 0.1, stop the whole train/predict process immediately
    # 1. If there is time budget limit, and some testing has already been done,
    #    but not enough remaining time for testing, then return None to stop
    # 2. Otherwise: make predictions normally, and update some
    #    variables for time managing
    if np.random.rand() < 0.01: # TODO: to change back to 0.1
      print_log("Oops! Choose to stop early!")
      self.done_training = True
      return None
    else:
      test_begin = time.time()
      if remaining_time_budget and self.estimated_time_test and\
          self.estimated_time_test > remaining_time_budget:
        print_log("Not enough time for test. "
              f"Estimated time for test: {self.estimated_time_test:.2e}, "
              f"But remaining time budget is: {remaining_time_budget:.2f}. "
              "Stop train/predict process by returning None.")
        return None
      msg_est = ""
      if self.estimated_time_test:
        msg_est = "estimated time: " + f"{self.estimated_time_test:.2e} sec."
      print_log(f"Begin testing...",msg_est)
      test_results = self.classifier.predict(input_fn=test_input_fn)
      predictions = [x['probabilities'] for x in test_results]
      predictions = np.array(predictions)
      test_end = time.time()
      test_duration = test_end - test_begin
      self.total_test_time += test_duration
      self.cumulated_num_tests += 1
      self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
      print_log(f"[+] Successfully made one prediction. {test_duration:.2f} sec used. "
            f"Total time used for testing: {self.total_test_time:.2f} sec. "
            f"Current estimated time for test: {self.estimated_time_test:.2e} sec.")
      return predictions

  def model_fn(self, features, labels, mode):
    """Model function to construct TensorFlow estimator.

    For details see:
    https://www.tensorflow.org/get_started/custom_estimators#write_a_model_function
    """

    col_count, row_count = self.metadata_.get_matrix_size(0)
    sequence_size = self.metadata_.get_sequence_size()
    output_dim = self.metadata_.get_output_size()

    # Sum over time axis
    input_layer = tf.reduce_sum(features['x'], axis=1)

    # Construct a neural network with 0 hidden layer
    input_layer = tf.reshape(input_layer,
                             [-1, row_count*col_count])

    # Replace missing values by 0
    input_layer = tf.where(tf.is_nan(input_layer),
                           tf.zeros_like(input_layer), input_layer)

    logits = tf.layers.dense(inputs=input_layer, units=output_dim)

    sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")
    softmax_tensor = tf.nn.softmax(logits, name="softmax_tensor")

    threshold = 0.5

    binary_predictions = tf.cast(tf.greater(sigmoid_tensor, threshold), tf.int32)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": binary_predictions,
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": softmax_tensor
      # "probabilities": sigmoid_tensor
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

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

def print_log(*content):
  """Logging function. (could've also used `import logging`.)"""
  now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
  print("MODEL INFO: " + str(now)+ " ", end='')
  print(*content)
