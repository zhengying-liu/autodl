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

It implements the 3 methods (i.e. __init__, train, test) described in
algorithm.py, which specifies the competition protocol in the comments.

It is EXTREMELY RECOMMENDED to look closely at this script algorithm.py in the
ingestion program folder (AutoDL_ingestion_program/), since any submission will
be consisted of a model.py script implementing a concrete class of
    algorithm.Algorithm
The submission can be done by uploading a zip file with model.py and an empty
file called `metadata`, like what we have in the folder
AutoDL_sample_code_submission/. And we also provide the option that a submission
can be a zip file zipping all CONTENT of AutoDL_starting_kit/.
"""

import tensorflow as tf
import algorithm
import os
import time
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

  #
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

    # Prediction made last time
    self.last_prediction = None

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.total_train_time = 0
    self.cumulated_num_steps = 0
    self.total_test_time = 0
    self.cumulated_num_tests = 0

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
    # 3. Choose a number (steps_to_train) randomly between 0 and max_steps and
    #    train for this many steps.
    if not self.cumulated_num_steps:
      steps_to_train = 1
    else:
      estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
      if self.cumulated_num_tests:
        estimated_time_test = self.total_test_time / self.cumulated_num_tests
      else:
        estimated_time_test = 0
      max_steps = int((remaining_time_budget - estimated_time_test) / estimated_time_per_step)
      # Choose random number of steps < max_steps for training
      steps_to_train = np.random.randint(0, max_steps)
    if steps_to_train == 0:
      time.sleep(remaining_time_budget) # Sleep 1 second if remaing time budget to small
    else:
      # Start training
      print(f"MODEL INFO: Begin training for {steps_to_train} steps...")
      train_start = time.time()
      with tf.Session() as sess:
        self.classifier.train(
          input_fn=train_input_fn,
          steps=steps_to_train)#,
          # hooks=[logging_hook])
      train_end = time.time()
      # Update for time budget managing
      train_duration = train_end - train_start
      self.total_train_time += train_duration
      self.cumulated_num_steps += steps_to_train
      estimated_time_per_step =\
          self.total_train_time / self.cumulated_num_steps
      print(f"MODEL INFO: {steps_to_train} steps trained. "
            f"Now total steps trained: {self.cumulated_num_steps}. "
            f"Total time used for training: {self.total_train_time:.2f}. "
            f"Current estimated time per step: {estimated_time_per_step:.2e}.")

  def test(self, dataset, remaining_time_budget=None):
    """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set.
    """
    # Turn `features` in the tensor pair (features, labels) to a dict
    dataset = dataset.map(lambda *x: ({'x': x[0]}, x[-1]))

    def test_input_fn():
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    # The following snippet of code intends to do:
    # 1. If there is time budget limit, and some testing has already been done,
    #    but not enough remaining time for testing, then return last prediction
    # 2. If there is time budget limit, and no testing has already been done,
    #    make an all-zero prediction
    # 3. In all other cases: make predictions normally, and update some
    #    variables for time managing
    if remaining_time_budget: # if there is time limit for predicting
      if self.cumulated_num_tests: # if some predictions are made previously
        estimated_time_test = self.total_test_time / self.cumulated_num_tests
        if estimated_time_test > remaining_time_budget: # if not enough time to make predictions
          print("Not enough time for test. Waiting for the training/predicting "
                "process to end and will use the last prediction...")
          time.sleep(remaining_time_budget)
          return self.last_prediction
      else:
        test_metadata = self.metadata_
        sample_count = test_metadata.size()
        output_dim = test_metadata.get_output_size()
        predictions = np.zeros((sample_count, output_dim))
        self.cumulated_num_tests += 1
        return predictions
    print("MODEL INFO: Begin testing...")
    test_begin = time.time()
    test_results = self.classifier.predict(input_fn=test_input_fn)
    test_end = time.time()
    test_duration = test_end - test_begin
    self.total_test_time += test_duration
    self.cumulated_num_tests += 1
    predictions = [x['probabilities'] for x in test_results] #TODO: make binary predictions
    predictions = np.array(predictions)
    self.last_prediction = predictions # Update last_prediction
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
