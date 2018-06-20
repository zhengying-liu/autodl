# Author: LIU Zhengying
# Creation Date: 18 June 2018
"""
A class to facilitate the management of datasets for AutoDL. It can keep
track of the files of each component (training set, test set, metadata, etc)
of a dataset, and make manipulations on them such as format transformation,
train/test split, example-label separation, checking, etc.

[IMPORTANT] To use this class, one should be very careful about file names! So
try not to modify file names manually.
Of course, originally this class is only reserved for organizers or their
collaborators, since the participants will only see the result of this class.
"""

import yaml
import os
import sys
# import tensorflow as tf
from tfrecord_utils import check_files_consistency

class DatasetManager(object):

  # Important YAML file assigned to save and load dataset info.
  # This file can be edited at first but should be automatically generated
  # once processed. So be really careful when using this file.
  global_filename = 'dataset_info.yaml'

  def __init__(self, dataset_dir, dataset_name=None):

    # Important (and compulsory) attributes
    self._dataset_info = {} # contains almost all useful info on this dataset
    self._dataset_dir = ""  # Absolute path to dataset directory
    self._dataset_name = ""

    if os.path.isdir(dataset_dir):
      self._dataset_dir = os.path.abspath(dataset_dir)
    else:
      raise ValueError(
        "Failed to create dataset manager. {} is not a directory!"\
                       .format(dataset_dir))

    self._path_to_yaml = os.path.join(self._dataset_dir,
                                      DatasetManager.global_filename)

    if dataset_name:
      self._dataset_name = dataset_name
    else:
      # Use folder name as dataset name
      self._dataset_name = os.path.basename(dataset_dir)

    if os.path.exists(self._path_to_yaml):
      self.load_dataset_info()
      # If loaded void YAML
      if not self._dataset_info:
        self.infer_dataset_info()
      if self._dataset_name != self._dataset_info['dataset_name']:
        print("WARNING: inconsistent dataset names!")
    else:
      self.infer_dataset_info()

  def save_dataset_info(self):
    with open(self._path_to_yaml, 'w') as f:
      print("Writing dataset info to the file {}."\
            .format(self._path_to_yaml))
      yaml.dump(self._dataset_info, f)
      print("Done!")

  def load_dataset_info(self):
    assert(os.path.exists(self._path_to_yaml))
    with open(self._path_to_yaml, 'r') as f:
      print("Loading dataset info with found file {}."\
            .format(self._path_to_yaml))
      self._dataset_info = yaml.load(f)
      print("Done!")

  def get_dataset_info(self):
    return self._dataset_info

  def get_default_dataset_info(self):
    default_dataset_info = {'dataset_name': self._dataset_name,
                            'metadata': None,
                            'training_data': {'examples': [],
                                              'labels': [],
                                              'format': 'tfrecord',
                                              'labels_separated': False
                                              },
                            'test_data': {'examples': [],
                                          'labels': [],
                                          'format': 'tfrecord',
                                          'labels_separated': True
                                          },
                            'consistency_check_done': False
                            }
    return default_dataset_info

  def infer_dataset_info(self):
    dataset_info = self.get_default_dataset_info()

    def is_sharded(path_to_tfrecord):
      return "-of-" in path_to_tfrecord

    files = os.listdir(self._dataset_dir)
    metadata_files = [x for x in files if 'metadata' in x]
    training_data_files = [x for x in files if 'train' in x]
    test_data_files = [x for x in files if 'test' in x]

    # Infer metadata
    if len(metadata_files) > 1:
      raise ValueError("More than 1 metadata files are found. Couldn't infer metadata.")
    elif len(metadata_files) < 1:
      dataset_info['metadata'] = None
    else:
      dataset_info['metadata'] = metadata_files[0]

    # Infer training data
    training_examples_files = [x for x in training_data_files if 'example' in x]
    training_labels_files = [x for x in training_data_files if 'label' in x]
    if len(training_labels_files) > 0:
      dataset_info['training_data']['labels_separated'] = True
      dataset_info['training_data']['examples'] = training_examples_files
      dataset_info['training_data']['labels'] = training_labels_files
    else:
      dataset_info['training_data']['labels_separated'] = False
      dataset_info['training_data']['examples'] = training_data_files
      dataset_info['training_data']['labels'] = training_data_files

    # Infer test data
    test_examples_files = [x for x in test_data_files if 'example' in x]
    test_labels_files = [x for x in test_data_files if 'label' in x]

    if test_labels_files: # if independent label files exist
      dataset_info['test_data']['labels_separated'] = True
      dataset_info['test_data']['examples'] = test_examples_files
      dataset_info['test_data']['labels'] = test_labels_files
    else:
      dataset_info['test_data']['labels_separated'] = False
      dataset_info['test_data']['examples'] = test_data_files
      dataset_info['test_data']['labels'] = test_data_files

    self._dataset_info = dataset_info


  def check_consistency(self):
    """Data Checker"""
    summary = {}
    for s in ['training_data', 'test_data']:
      summary[s + '_format'] = self._dataset_info[s]['format']
      for t in ['examples', 'labels']:
        abspaths = [os.path.join(self._dataset_dir, x)
                    for x in self._dataset_info[s][t]]
        n, c, f = check_files_consistency(abspaths)
        summary['num_' + t + '_of_' + s] = n
        summary['context_' + t + '_of_' + s] = c
        summary['feature_lists_' + t + '_of_' + s] = f

    # Check that num examples and num labels are consistent
    for s in ['training_data', 'test_data']:
      assert(summary['num_examples_of_' + s] == summary['num_labels_of_' + s])

    print("Consistency check done for the dataset {}, here's the summary: "\
          .format(self._dataset_name))
    from pprint import pprint
    pprint(summary)
    self._dataset_info['consistency_check_done'] = True


  def convert_AutoML_to_AutoDL(self, *arg, **kwarg):
    """Convert a dataset in AutoML format to AutoDL format.

    This facilitates the process of generating new datasets in AutoDL format,
    since there exists a big database of datasets in AutoML format.
    """
    pass

  def train_test_split(self):
    """Split the dataset to have training data and test data
    """
    pass

  def remove_all_irrelevant_files_in_dataset_dir(self):
    pass

def main():
  pass

if __name__ == '__main__':
  main()
