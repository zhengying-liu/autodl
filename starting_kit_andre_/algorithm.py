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

"""Class for supervised machine learning algorithms for the autodl project.

This is the API; see algorithm.py, algorithm_scikit.py for implementations.
"""

class Algorithm(object):
  """Algorithm class: API (abstract class)."""

  def __init__(self, metadata):
    self.metadata_ = metadata

  def train_by_time(self, dataset, max_time):
    del max_time
    return self.train(dataset)

  def train(self, dataset):
    """Train this algorithm on the tensorflow |dataset|."""
    raise NotImplementedError("Algorithm class does not have any training.")

  def predict(self, *input_arg):
    """Get the output of this algorithm on a single input (list of matrices).

    Args:
      *input_arg: variable list of input matrices, one argument is one matrix.
        The number of matrices is given by metadata.get_bundle_size() and the
        size of each argument is given by metadata.get_matrix_size(arg_index).

    Returns:
      * a 1D array encoding the multi-class output of this algorithm on
        input_arg.
    """
    raise NotImplementedError("Algorithm class does not have any testing.")
